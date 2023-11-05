
# Import libraries
import torch
import torch.nn as nn
from losses.triplet_loss import CrossEntropyLabelSmooth
import pytorch_lightning as pl
from main_model import ReidMainModel
from torch import optim
from optim import lr_scheduler


### Load the main model
class ReidCatModel(pl.LightningModule):
    def __init__(self, backbone_model,number_classes, embedding_size,steps_main_opt,lr_main = 3e-4,arcface = True, num_epochs = 50,sch_gamma = 0.5, sch_warmup_factor = 0.01,sch_warmup_iter = 25):
        super(ReidCatModel, self).__init__()   
        
        # Set variables
        self.backbone_model = backbone_model
        self.number_classes = number_classes
        self.embedding_size = embedding_size
        self.steps_main_opt = steps_main_opt
        self.sch_gamma = sch_gamma
        self.sch_warmup_factor = sch_warmup_factor
        self.sch_warmup_iter = sch_warmup_iter
        self.lr_main = lr_main
        self.arcface = arcface
        self.num_epochs = num_epochs
        
        # Define manual optimization
        self.automatic_optimization = False

        # Define main losses
        self.ce_loss = CrossEntropyLabelSmooth(num_classes=self.number_classes)
        
        # Define historic metric for training
        self.batch_train_loss = []
        self.batch_train_acc = []
        self.hist_train_loss = []
        self.hist_train_acc = []
        
        # Historic for validation
        self.batch_val_loss = []
        self.batch_val_acc = []
        self.hist_val_loss = []
        self.hist_val_acc = []

        # Import the model
        self.model = ReidMainModel(backbone_model = self.backbone_model,number_classes = self.number_classes, embedding_size = self.embedding_size,arcface = self.arcface)
        
    # Define forward step
    def forward(self, x, label = None):
        classify, embedding = self.model.forward(x,label)
        return classify, embedding
    
    # Training step
    def training_step(self,batch,batch_idx):
        # Defining optimizer
        optimizer = self.optimizers()

        image, label, trunk, left_leg, right_leg, left_thig, right_thig, left_shank, right_shank, front_tail, rear_tail = batch
        # Forward pass
        logits_full, logits_trunk, logits_limbs, embedding, trunk_features, limbs_features  = self.model(image, trunk, left_leg, right_leg, left_thig, right_thig, left_shank, right_shank,front_tail,rear_tail, label)

        # Using best model
        id_g, id_gb, id_gp, triplet_gb, triplet_gp = loss_fn(logits_full, logits_trunk, logits_limbs, trunk_features, limbs_features, label)
        train_loss = id_g + 1.5 *id_gb + 1.5 * id_gp + 2 * triplet_gb + 2 * triplet_gp
        
        # Run main optimizer
        optimizer.zero_grad()
        self.manual_backward(train_loss)
        optimizer.step()

        # Calculate acc
        train_acc = self.calculate_topk_accuracy(logits_full,label)
        # Save the loss and acc values
        self.batch_train_loss.append(triplet_gb)
        self.batch_train_acc.append(train_acc)
        self.log_dict({'train_loss': train_loss, 'train_acc': train_acc})
        

    # Validation step
    def validation_step(self,batch, batch_idx):
        image, label, imgid = batch
        # Forward pass
        logits, embedding  = self.forward(image,label)
        # Calculate loss function
        val_loss = self.ce_loss(logits,label)
        # Calculate acc
        val_acc = self.calculate_topk_accuracy(logits,label)
        # Save the loss and acc values
        self.batch_val_loss.append(val_loss)
        self.batch_val_acc.append(val_acc)
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc})
        return {'loss': val_loss, 'acc': val_acc}

    # Process on final epoch
    def on_train_epoch_end(self):
        # Get the loss mean
        loss_train_epoch_mean = torch.stack(self.batch_train_loss).mean()
        self.hist_train_loss.append(loss_train_epoch_mean.cpu().detach().numpy())
        # free up the memory
        self.batch_train_loss.clear()

        # Get the acc mean
        training_acc = [acc for acc in self.batch_train_acc]
        self.hist_train_acc.append(sum(training_acc)/len(training_acc))
        # free up the memory
        self.batch_train_acc.clear()

    # Process on final epoch
    def on_validation_epoch_end(self):
        # Get the loss mean
        loss_val_epoch_mean = torch.stack(self.batch_val_loss).mean()
        self.hist_val_loss.append(loss_val_epoch_mean.cpu().detach().numpy())
        # free up the memory
        self.batch_val_loss.clear()

        # Get the acc mean
        validation_acc = [acc for acc in self.batch_val_acc]
        self.hist_val_acc.append(sum(validation_acc)/len(validation_acc))
        # free up the memory
        self.batch_val_acc.clear()

    # Prediction
    def on_predict_start(self):
        # Create outputs
        self.pred_embeddings = []
        self.pred_logits = []
        self.pred_img_id = []

    def predict_step(self, batch, batch_idx):
        images, id_class, imgid = batch
        logits, embeddings = self.forward(images)
        # Saving outputs
        self.pred_embeddings.append(embeddings)
        self.pred_logits.append(logits)
        self.pred_img_id.append(imgid)
        # Return 
        return embeddings, logits, imgid

    def on_predict_epoch_end(self):
        self.pred_embeddings = torch.cat(self.pred_embeddings, dim = 0)
        self.pred_logits = torch.cat(self.pred_logits, dim = 0)
        self.pred_img_id = torch.cat(self.pred_img_id, dim = 0).cpu().numpy()

    # Define accuracy metric
    def calculate_topk_accuracy(self, y_pred, y_true, k=1):
        topk_preds = torch.topk(y_pred, k=k, dim=1)[1]
        y_true_expanded = y_true.unsqueeze(1).expand_as(topk_preds)
        correct = torch.sum(torch.any(topk_preds == y_true_expanded, dim=1)).item()
        total = len(y_true)
        accuracy = correct / total
        return accuracy
    
    # Configure optimizer
    def configure_optimizers(self):
        # Define optimizer
        optimizer = optim.Adam(self.parameters(),lr = self.lr_main, weight_decay =  0.0005)
        scheduler = lr_scheduler.WarmupMultiStepLR(optimizer,self.steps_main_opt, self.sch_gamma,self.sch_warmup_factor,self.sch_warmup_iter)
        return [optimizer], [scheduler]

    # Record metric
    def recorded_metrics(self):
        # We ignore the validation metrics within the sanity check
        return self.hist_train_loss, self.hist_train_acc, self.hist_val_loss[1: ], self.hist_val_acc[1:]
