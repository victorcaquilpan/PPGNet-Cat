# Import libraries
import numpy as np

# Create a cosine similarity function
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    return dot_product / (norm_a * norm_b)