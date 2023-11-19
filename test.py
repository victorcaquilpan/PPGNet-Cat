# Import libraries
import torch
import numpy as np
from pytorch_lightning import Trainer
import json
from argparse import ArgumentParser

# Import other complementary scripts
from models.pl_inference import ReidPrediction
from datasets.dataloader import ReidDataModule
from utils.cosine_similarity import cosine_similarity
from utils.re_ranking import re_ranking
from utils.evaluation import evaluate

def main(args):
    #Parameteres
    class Config():
        CAT_TESTING_DIR = args.cat_testing_dir
        CAT_ANNO_TEST_FILE = args.cat_anno_test_file
        EVALUATION_FILE = args.evaluation_file
        NUMBER_WORKERS = args.number_workers
        NUM_CLASSES = args.num_classes
        BATCH_SIZE_TEST = args.batch_size_test
        SIZE_FULL_IMAGE = args.size_full_image
        EMBEDDING_SIZE = args.embedding_size
        ARCFACE = args.arcface
        BACKBONE = args.backbone
        DETERMINISTIC = args.deterministic
        PRECISION = args.precision
        TRAINED_MODEL = args.trained_model

    # Creating dataloader
    cat_data = ReidDataModule(data_directory=Config(),
                                batch_size_test = Config().BATCH_SIZE_TEST,
                                transform= False,
                                num_workers= Config().NUMBER_WORKERS, 
                                size_full_image = Config().SIZE_FULL_IMAGE)
    # Call the setup method
    cat_data.setup()

    # Create the model
    eval_model = ReidPrediction(
        backbone_model = Config().BACKBONE,
        number_classes = Config().NUM_CLASSES, 
        embedding_size = Config().EMBEDDING_SIZE,
        arcface = Config().ARCFACE)

    # Create the trainer
    trainer = Trainer(accelerator='gpu', logger = False, 
                    enable_checkpointing=False, 
                    precision = Config().PRECISION, 
                    deterministic = Config().DETERMINISTIC)

    # Load the weights and biases
    eval_model.full_image_model.load_state_dict(torch.load('pretrained_weights/' + Config().TRAINED_MODEL))

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
    print(evaluate(Config().EVALUATION_FILE,'cat_results',phase_codename='dev'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cat_testing_dir", type=str, default='data/test/images/',
                        help="Path of cat images for testing")
    parser.add_argument("--cat_anno_test_file", type=str, default='data/test/test_anno.csv',
                        help="Filepath of annotations for testing images")
    parser.add_argument("--evaluation_file", type=str, default='data/test/gt_test_plain.json',
                        help="Filepath of ground truth")
    parser.add_argument("--number_workers", type=int, default = 8)
    parser.add_argument("--num_classes", type = int, default = 300)
    parser.add_argument("--batch_size_test",type = int, default = 2)
    parser.add_argument("--size_full_image", type = tuple, default = (256,512))
    parser.add_argument("--embedding_size", type = int, default = 2560)
    parser.add_argument("--arcface", type = bool, default = False)
    parser.add_argument("--backbone", type = str, default= "resnet152")
    parser.add_argument("--deterministic", type = list, default = [True, "warn"])
    parser.add_argument("--precision", type = str, default = "16-mixed")
    parser.add_argument("--trained_model", default = "best_model.pth", help = "Use of weights")
    args = parser.parse_args()

main(args)       

