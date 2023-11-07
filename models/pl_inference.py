# Import libraries
import pytorch_lightning as pl
import torch
from models.reid_models import ReidModel


### Load the main model
class ReidPrediction(pl.LightningModule):
    def __init__(self, backbone_model,number_classes, embedding_size,arcface = True):
        super(ReidPrediction, self).__init__()   
        
        # Set variables
        self.backbone_model = backbone_model
        self.number_classes = number_classes
        self.embedding_size = embedding_size
        self.arcface = arcface
    
        # Define the model
        self.full_image_model = ReidModel(backbone_model=self.backbone_model,number_classes = self.number_classes,embedding_size = self.embedding_size, arcface = self.arcface)

    # Define forward step
    def forward(self, x, label = None):
        classify, embedding = self.full_image_model(x,label)
        return classify, embedding
    
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
