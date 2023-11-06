
# Load basic libraries
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pandas as pd
import json
from pytorch_lightning.callbacks import ModelCheckpoint

# Complementary scripts
from utils.evaluation import evaluate
from utils.re_ranking import re_ranking
from utils.cosine_similarity import cosine_similarity
from losses import make_loss_with_parameters
from datasets.dataloader import ReidDataModule
from models.pl_model import ReidCatModel

# Set seed
torch.manual_seed(1) 

#Paramteres
class Config():
    cat_training_dir = 'data/train/images/'  
    cat_anno_train_file = 'data/train/train_anno.csv'
    keypoints_train = 'data/train/keypoints_train.csv'

    cat_testing_dir = 'data/test/images/'
    #cat_testing_dir = 'data/feral_cat/melbourne/reid_images/'
    #cat_testing_dir = 'data/feral_cat/test_fusion/'
    
    cat_anno_test_file = 'data/test/test_anno.csv'
    #cat_anno_test_file = 'data/feral_cat/melbourne/anno_test_data.csv'
    #cat_anno_test_file = 'data/feral_cat/anno_cat_combined_data.csv'
    
    evaluation_file = 'data/test/gt_test_plain.json'
    #evaluation_file = 'data/feral_cat/melbourne/gt_test_plain.json'
    #evaluation_file = 'data/feral_cat/gt_test_plain_combined.json'

    number_workers = 8
    batch_size_train = 18 # 18 
    batch_size_test = 2
    number_epochs = 2
    save_n_epochs = 50
    transformation = True
    size_full_image = (256,512)
    size_trunk_image = (64,128)
    size_limb_image = (64,64)
    steps_main_opt = [40, 80, 120, 160, 240, 320, 400]
    sch_gamma = 0.5
    sch_warmup_factor = 0.01
    sch_warmup_iter = 25    
    num_classes = 100
    embeddings = 2560
    arcface = False
    lr_main = 0.00025
    backbone = 'resnet152'
    deterministic = [True, "warn"]
    precision = "16-mixed"
    mirrowed_data = True
    include_cat_keypoints = True
    base_model = None
    retrain = True


# Creating dataloader
tiger_data = ReidDataModule(data_directory=Config(),
                            batch_size_train = Config().batch_size_train,
                            batch_size_test = Config().batch_size_test,
                            transform=Config().transformation,
                            num_workers= Config().number_workers, 
                            size_full_image = Config().size_full_image,
                            size_trunk_image = Config().size_trunk_image,
                            size_limb_image = Config.size_limb_image,
                            mirrowed_images = Config().mirrowed_data,
                            include_cat_keypoints=Config().include_cat_keypoints)
# Call the setup method
tiger_data.setup()


# Create our main loss function
loss_fn = make_loss_with_parameters(Config().num_classes)

# Create the main model
if Config().base_model != None:

    # Load the parameters from a previous implementation
    model = ReidCatModel.load_from_checkpoint('pretrained_weights/' + Config().base_model, 
                                        backbone_model = Config().backbone,
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
else:
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


# Create a LearningRateMonitor callback
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

# Create checkpoints
if Config().arcface:
    arcface = 'arcface_'
else:
    arcface = ''

# Define the checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath='pretrained_weights/',
    filename= arcface + Config().backbone + '_testing-{epoch:02d}',
    every_n_epochs = Config().save_n_epochs,
    save_weights_only=True,
    save_top_k = -1)

# Define the Trainer
trainer = Trainer(max_epochs=Config().number_epochs,
                accelerator='gpu', logger = True, 
                enable_checkpointing=True, 
                callbacks=[lr_monitor,checkpoint_callback],
                precision = Config().precision, 
                deterministic = Config().deterministic)


if Config.retrain:
    # Training
    trainer.fit(model = model,
            train_dataloaders=tiger_data.train_dataloader(),
            val_dataloaders=tiger_data.val_dataloader())

# Evaluation of model
predictions = trainer.predict(model,dataloaders=tiger_data.test_dataloader())
 
# Create a empty matrix
dist_matrix = np.zeros((len(model.pred_img_id),len(model.pred_img_id)))

for query_im in range(0, len(model.pred_img_id)):
    for collection_im in range(0,len(model.pred_img_id)):
        dist_matrix[query_im,collection_im] = cosine_similarity(model.pred_embeddings[query_im].cpu().numpy(), model.pred_embeddings[collection_im].cpu().numpy())

# Get the reranked distance matrix
reranked_dist = re_ranking(dist_matrix, dist_matrix, dist_matrix, k1=20, k2=6, lambda_value=0.3)

# Create output list
prediction_cat_results = []

# Calculate the distance between the query image and each one the the remaining images
for query_im in range(0, len(model.pred_img_id)):
    # Create a dictionary
    dict_query = {"query_id": int(model.pred_img_id[query_im])}
    # Save the results in a list
    result_query = []
    # Calculate the distance over the remaining images
    for ans_img in range(0, len(model.pred_img_id)):
        if query_im != ans_img:
            img_dist = (int(model.pred_img_id[ans_img]) , reranked_dist[query_im,ans_img])
            result_query.append(img_dist)
    # Sort images
    result_query = sorted(result_query, key=lambda x: x[1],reverse=False)
    result_query = [img[0] for img in result_query]
    dict_query["ans_ids"] = result_query
    prediction_cat_results.append(dict_query)

# Save the list of dictionaries to the JSON file
with open('cat_results', "w") as json_file:
    json.dump(prediction_cat_results, json_file, indent=4)

# Evaluation
print(evaluate(Config().evaluation_file,'cat_results',phase_codename='dev'))