
# train_model_svd.py

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import pickle

class RecommenderTrainerSVD:
    def __init__(self, user_item_matrix, num_factors=50):
        self.user_item_matrix = user_item_matrix
        self.num_factors = num_factors
        self.model = None

    def train_model(self):
        """
        Train a collaborative filtering model using SVD (Singular Value Decomposition).
        """
        # Corrected mean calculation with .values to ensure correct reshaping
        user_ratings_mean = np.mean(self.user_item_matrix.values, axis=1)
        user_item_matrix_demeaned = self.user_item_matrix.values - user_ratings_mean.reshape(-1, 1)
        
        # Perform matrix factorization
        U, sigma, Vt = svds(user_item_matrix_demeaned, k=self.num_factors)
        sigma = np.diag(sigma)
        
        # Reconstruct predictions using the decomposed matrices
        self.model = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        print("SVD model training completed successfully.")
        return self.model

    def save_model(self, filepath='recommender_svd_model.pkl'):
        """
        Save the trained SVD model to a file.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"SVD model saved to {filepath} successfully.")

    def retrain_model(self, user_item_matrix):
        """
        Retrain the SVD model with updated user-item matrix.
        """
        print("Retraining SVD model...")
        self.user_item_matrix = user_item_matrix
        return self.train_model()