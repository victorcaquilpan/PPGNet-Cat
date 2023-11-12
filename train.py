# Load basic libraries
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Complementary scripts
from losses import make_loss_with_parameters
from datasets.dataloader import ReidDataModule
from models.pl_model import ReidCatModel

# Set seed
torch.manual_seed(123) 

#Parameteres
class Config():
    cat_training_dir = 'data/train/images/'  
    cat_anno_train_file = 'data/train/train_anno.csv'
    keypoints_train = 'data/train/keypoints_train.csv'

    # cat_training_dir = 'data/tiger/train/images/'
    # cat_anno_train_file = 'data/tiger/train/reid_list_train.csv'
    # keypoints_train = 'data/tiger/train/reid_keypoints_train.json'

    number_workers = 8
    batch_size_train = 22
    batch_size_val = 2
    number_epochs = 300
    transformation = True
    size_full_image = (256,512)
    size_trunk_image = (64,128)
    size_limb_image = (64,64)
    steps_main_opt = [40, 80, 120, 160, 240, 320, 400]
    sch_gamma = 0.5
    sch_warmup_factor = 0.01
    sch_warmup_iter = 25    
    num_classes = 600
    embeddings = 2560
    arcface = False
    lr_main = 0.00025
    backbone = 'resnet152'
    deterministic = [True, "warn"]
    precision = "16-mixed"
    mirrored_data = True
    include_cat_keypoints = True
    min_images_per_entity = 8
    base_model = None
    retrain = True

# Creating dataloader
cat_data = ReidDataModule(data_directory=Config(),
                            batch_size_train = Config().batch_size_train,
                            batch_size_val = Config().batch_size_val,
                            transform=Config().transformation,
                            num_workers= Config().number_workers, 
                            size_full_image = Config().size_full_image,
                            size_trunk_image = Config().size_trunk_image,
                            size_limb_image = Config.size_limb_image,
                            mirrored_images = Config().mirrored_data,
                            include_cat_keypoints=Config().include_cat_keypoints,
                            min_images_per_entity = Config().min_images_per_entity)
# Call the setup method
cat_data.setup()


# Create our main loss function
loss_fn = make_loss_with_parameters(Config().num_classes)


# Create a model
model = ReidCatModel(backbone_model= Config().backbone,
                number_classes=Config().num_classes, 
                embedding_size = Config().embeddings, 
                main_loss = loss_fn,
                steps_main_opt=Config().steps_main_opt,
                lr_main = Config().lr_main,
                arcface = Config().arcface,
                num_epochs = Config().number_epochs,
                sch_gamma = Config().sch_gamma,
                sch_warmup_factor = Config().sch_warmup_factor,
                sch_warmup_iter = Config().sch_warmup_iter)

if Config().base_model:
    # Load the weights and biases
    model.load_state_dict(torch.load('pretrained_weights/' + Config().base_model))
    print('loaded')


# Create a LearningRateMonitor callback
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

# Define the Trainer
trainer = Trainer(max_epochs=Config().number_epochs,
                accelerator='gpu', logger = True, 
                enable_checkpointing=True, 
                callbacks=[lr_monitor],
                precision = Config().precision, 
                deterministic = Config().deterministic)

if Config.retrain:
    # Training
    trainer.fit(model = model,
            train_dataloaders=cat_data.train_dataloader(),
            val_dataloaders=cat_data.val_dataloader())

# Save the weights and biases    
torch.save(model.model.full_image_model.state_dict(),'pretrained_weights/eval_model.pth')
torch.save(model.state_dict(),'pretrained_weights/full_model.pth')


# Printing message
print('Training was done successfully! Model was saved in "pretrained weights directory"')
    

