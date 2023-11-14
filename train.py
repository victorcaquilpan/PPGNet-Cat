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
    CAT_TRAINING_DIR = 'data/train/images/' 
    CAT_ANNO_TRAIN_FILE = 'data/train/train_anno.csv'
    KEYPOINTS_TRAIN = 'data/train/keypoints_train.csv'
    NUMBER_WORKERS = 8
    BATCH_SIZE_TRAIN = 22
    BATCH_SIZE_VAL = 2
    NUMBER_EPOCHS = 500
    TRANSFORMATION = True
    SIZE_FULL_IMAGE = (256,512)
    SIZE_TRUNK_IMAGE = (64,128)
    SIZE_LIMB_IMAGE = (64,64)
    STEPS_MAIN_OPT = [40, 80, 120, 160, 240, 320, 400] 
    SCH_GAMMA = 0.5
    SCH_WARMUP_FACTOR = 0.01
    SCH_WARMUP_ITER = 25
    NUM_CLASSES = 600
    EMBEDDING_SIZE = 2560
    ARCFACE = False
    LR_MAIN = 0.00025
    BACKBONE = 'resnet152'
    DETERMINISTIC = [True, "warn"]
    PRECISION = "16-mixed"
    MIRRORED_DATA = True
    INCLUDE_CAT_KEYPOINTS = True
    MIN_IMAGES_PER_ENTITY = 8
    BASE_MODEL = None
    RETRAIN = True

# Creating dataloader
cat_data = ReidDataModule(data_directory=Config(),
                            batch_size_train = Config().BATCH_SIZE_TRAIN,
                            batch_size_val = Config().BATCH_SIZE_VAL,
                            transform=Config().TRANSFORMATION,
                            num_workers= Config().NUMBER_WORKERS, 
                            size_full_image = Config().SIZE_FULL_IMAGE,
                            size_trunk_image = Config().SIZE_TRUNK_IMAGE,
                            size_limb_image = Config.SIZE_LIMB_IMAGE,
                            mirrored_images = Config().MIRRORED_DATA,
                            include_cat_keypoints=Config().INCLUDE_CAT_KEYPOINTS,
                            min_images_per_entity = Config().MIN_IMAGES_PER_ENTITY)
# Call the setup method
cat_data.setup()

# Create our main loss function
loss_fn = make_loss_with_parameters(Config().NUM_CLASSES)

# Create a model
model = ReidCatModel(backbone_model= Config().BACKBONE,
                number_classes=Config().NUM_CLASSES, 
                embedding_size = Config().EMBEDDING_SIZE, 
                main_loss = loss_fn,
                steps_main_opt=Config().STEPS_MAIN_OPT,
                lr_main = Config().LR_MAIN,
                arcface = Config().ARCFACE,
                num_epochs = Config().NUMBER_EPOCHS,
                sch_gamma = Config().SCH_GAMMA,
                sch_warmup_factor = Config().SCH_WARMUP_FACTOR,
                sch_warmup_iter = Config().SCH_WARMUP_ITER)

if Config().BASE_MODEL:
    # Load the weights and biases
    model.load_state_dict(torch.load('pretrained_weights/' + Config().BASE_MODEL))
    print('loaded')


# Create a LearningRateMonitor callback
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

# Define the Trainer
trainer = Trainer(max_epochs=Config().NUMBER_EPOCHS,
                accelerator='gpu', logger = True, 
                enable_checkpointing=True, 
                callbacks=[lr_monitor],
                precision = Config().PRECISION, 
                deterministic = Config().DETERMINISTIC)

if Config.RETRAIN:
    # Training
    trainer.fit(model = model,
            train_dataloaders=cat_data.train_dataloader(),
            val_dataloaders=cat_data.val_dataloader())

# Save the weights and biases    
torch.save(model.model.full_image_model.state_dict(),'pretrained_weights/eval_model.pth')

# Printing message
print('Training was done successfully! Model was saved in "pretrained weights directory"')
    

