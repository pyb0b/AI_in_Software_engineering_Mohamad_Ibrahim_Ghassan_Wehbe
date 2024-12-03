
# train_model.py

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

class RecommenderTrainerKNN:
    def __init__(self, user_item_matrix, metric='cosine', algorithm='brute'):
        self.user_item_matrix = user_item_matrix
        self.metric = metric
        self.algorithm = algorithm
        self.model = None

    def train_model(self):
        """
        Train a collaborative filtering model using k-Nearest Neighbors (k-NN) with cosine similarity.
        """
        # Convert user-item matrix to sparse format
        sparse_user_item_matrix = csr_matrix(self.user_item_matrix.values)

        # Initialize and fit the k-NN model
        self.model = NearestNeighbors(metric=self.metric, algorithm=self.algorithm)
        self.model.fit(sparse_user_item_matrix)
        print("k-NN model training completed successfully.")
        return self.model

    def save_model(self, filepath='recommender_knn_model.pkl'):
        """
        Save the trained k-NN model to a file.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {filepath} successfully.")
        
    def retrain_model(self, user_item_matrix):
        """
        Retrain the k-NN model with updated user-item matrix.
        """
        print("Retraining k-NN model...")
        self.user_item_matrix = user_item_matrix
        return self.train_model()
