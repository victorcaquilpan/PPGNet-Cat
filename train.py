# Load basic libraries
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser

# Complementary scripts
from losses import make_loss_with_parameters
from datasets.dataloader import ReidDataModule
from models.pl_model import ReidCatModel

# Set seed
torch.manual_seed(123) 

def main(args):

    #Parameteres
    class Config():
        CAT_TRAINING_DIR = args.cat_training_dir
        CAT_ANNO_TRAIN_FILE = args.cat_anno_train_file
        KEYPOINTS_TRAIN = args.keypoints_train
        NUMBER_WORKERS = args.number_workers
        BATCH_SIZE_TRAIN = args.batch_size_train
        BATCH_SIZE_VAL = args.batch_size_val
        NUMBER_EPOCHS = args.number_epochs
        TRANSFORMATION = args.transformation
        SIZE_FULL_IMAGE = args.size_full_image
        SIZE_TRUNK_IMAGE = args.size_trunk_image
        SIZE_LIMB_IMAGE = args.size_limb_image
        NUM_CLASSES = args.num_classes
        EMBEDDING_SIZE = args.embedding_size
        ARCFACE = args.arcface
        BACKBONE = args.backbone
        DETERMINISTIC = args.deterministic
        PRECISION = args.precision
        MIRRORED_DATA = args.mirrored_data
        MIN_IMAGES_PER_ENTITY = args.min_images_per_entity
        BASE_MODEL = args.base_model
        NAME_OUTPUT_MODEL = args.name_output_model
        RETRAIN = args.retrain
        # Other parameters
        INCLUDE_CAT_KEYPOINTS = True
        LR_MAIN = 0.00025
        STEPS_MAIN_OPT = [40, 80, 120, 160, 240, 320, 400] 
        SCH_GAMMA = 0.5
        SCH_WARMUP_FACTOR = 0.01
        SCH_WARMUP_ITER = 25

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
    torch.save(model.model.full_image_model.state_dict(),'pretrained_weights/' + Config().NAME_OUTPUT_MODEL)

    # Printing message
    print('Training was done successfully! Model was saved in "pretrained weights directory"')
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cat_training_dir", type=str, default='data/train/images/',
                        help="Path of cat images for training")
    parser.add_argument("--cat_anno_train_file", type=str, default='data/train/train_anno.csv',
                        help="Filepath of annotations for training images")
    parser.add_argument("--keypoints_train", type=str, default='data/train/keypoints_train.csv',
                        help="Filepath of keypoints")
    parser.add_argument("--number_workers", type=int, default = 8)
    parser.add_argument("--batch_size_train",type = int, default=22)
    parser.add_argument("--batch_size_val",type = int, default = 2)
    parser.add_argument("--number_epochs", type = int, default = 200)
    parser.add_argument("--transformation", type = bool, default= True,help = "Incorporate data augmentation")
    parser.add_argument("--size_full_image", type = tuple, default = (256,512))
    parser.add_argument("--size_trunk_image", type = tuple, default = (64,128))
    parser.add_argument("--size_limb_image", type = tuple, default = (64,64))
    parser.add_argument("--num_classes", type = int, default = 300)
    parser.add_argument("--embedding_size", type = int, default = 2560)
    parser.add_argument("--arcface", type = bool, default = False)
    parser.add_argument("--backbone", type = str, default= "resnet152")
    parser.add_argument("--deterministic", type = list, default = [True, "warn"])
    parser.add_argument("--precision", type = str, default = "16-mixed")
    parser.add_argument("--mirrored_data", type = bool, default = True, help = "Flip images to create new entities")
    parser.add_argument("--min_images_per_entity", type = int, default = 8, help = "Ensuring a min of images per entity")
    parser.add_argument("--base_model", default = None, help = "Use of a base model")
    parser.add_argument("--name_output_model", default= 'eval_model.pth', help = "Name of trained weights")
    parser.add_argument("--retrain", type = bool,default = True) 

    args = parser.parse_args()

main(args)       

