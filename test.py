# Import libraries
import torch
import numpy as np
from pytorch_lightning import Trainer
import json

# Import other complementary scripts
from models.pl_inference import ReidPrediction
from datasets.dataloader import ReidDataModule
from utils.cosine_similarity import cosine_similarity
from utils.re_ranking import re_ranking
from utils.evaluation import evaluate

#Parameteres
class Config():
    cat_testing_dir = 'data/test/images/'
    cat_anno_test_file = 'data/test/test_anno.csv'
    evaluation_file = 'data/test/gt_test_plain.json'
    number_workers = 8
    num_classes = 100
    batch_size_test = 2
    transformation = True
    size_full_image = (256,512)
    embeddings = 2560
    arcface = False
    backbone = 'resnet152'
    deterministic = [True, "warn"]
    precision = "16-mixed"
    trained_model = 'best_model.pth'


# Creating dataloader
cat_data = ReidDataModule(data_directory=Config(),
                            batch_size_test = Config().batch_size_test,
                            transform=Config().transformation,
                            num_workers= Config().number_workers, 
                            size_full_image = Config().size_full_image)
# Call the setup method
cat_data.setup()

# Create the model
eval_model = ReidPrediction(
    backbone_model = Config().backbone,
    number_classes = Config().num_classes, 
    embedding_size = Config().embeddings,
    arcface = Config().arcface)

# Create the trainer
trainer = Trainer(accelerator='gpu', logger = False, 
                enable_checkpointing=False, 
                precision = Config().precision, 
                deterministic = Config().deterministic)

# Load the weights and biases
eval_model.full_image_model.load_state_dict(torch.load('pretrained_weights/' + Config().trained_model))

# # Evaluation of model
predictions = trainer.predict(eval_model,dataloaders=cat_data.test_dataloader())

# Re-ranking
 
# Create a empty matrix
dist_matrix = np.zeros((len(eval_model.pred_img_id),len(eval_model.pred_img_id)))

for query_im in range(0, len(eval_model.pred_img_id)):
    for collection_im in range(0,len(eval_model.pred_img_id)):
        dist_matrix[query_im,collection_im] = cosine_similarity(eval_model.pred_embeddings[query_im].cpu().numpy(), eval_model.pred_embeddings[collection_im].cpu().numpy())

# Get the reranked distance matrix
reranked_dist = re_ranking(dist_matrix, dist_matrix, dist_matrix, k1=20, k2=6, lambda_value=0.3)

# Create output list
prediction_cat_results = []

# Calculate the distance between the query image and each one the the remaining images
for query_im in range(0, len(eval_model.pred_img_id)):
    # Create a dictionary
    dict_query = {"query_id": int(eval_model.pred_img_id[query_im])}
    # Save the results in a list
    result_query = []
    # Calculate the distance over the remaining images
    for ans_img in range(0, len(eval_model.pred_img_id)):
        if query_im != ans_img:
            img_dist = (int(eval_model.pred_img_id[ans_img]) , reranked_dist[query_im,ans_img])
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

