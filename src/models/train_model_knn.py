import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle
from pandas import DataFrame
from typing import Optional

import mlflow
import mlflow.sklearn

class RecommenderTrainerKNN:
    """
    A class to train and manage a k-Nearest Neighbors (k-NN) model for collaborative filtering.
    """

    def __init__(
        self, 
        user_item_matrix: DataFrame, 
        metric: str = "cosine", 
        algorithm: str = "brute"
    ) -> None:
        """
        Initialize the k-NN recommender trainer.

        Parameters:
        - user_item_matrix (DataFrame): User-item interaction matrix.
        - metric (str): Distance metric to use (default: "cosine").
        - algorithm (str): Algorithm to use for nearest neighbors (default: "brute").
        """
        self.user_item_matrix = user_item_matrix
        self.metric = metric
        self.algorithm = algorithm
        self.model: Optional[NearestNeighbors] = None

    def train_model(self) -> NearestNeighbors:
        """
        Train a collaborative filtering model using k-Nearest Neighbors (k-NN) with the specified metric.

        Returns:
        - NearestNeighbors: Trained k-NN model.
        """
        # Convert user-item matrix to sparse format
        sparse_user_item_matrix = csr_matrix(self.user_item_matrix.values)
        
        with mlflow.start_run(run_name="kNN Model Training"):
            # Log parameters
            mlflow.log_param("metric", self.metric)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("matrix_shape", sparse_user_item_matrix.shape)
            
            # Initialize and fit the k-NN model
            self.model = NearestNeighbors(metric=self.metric, algorithm=self.algorithm)
            self.model.fit(sparse_user_item_matrix)
            
            # Log the model as an artifact
            model_path = "recommender_knn_model.pkl"
            with open(model_path, "wb") as file:
                pickle.dump(self.model, file)
            mlflow.log_artifact(model_path)

            print("k-NN model training completed successfully.")
            return self.model

    def save_model(self, filepath: str = "recommender_knn_model.pkl") -> None:
        """
        Save the trained k-NN model to a file.

        Parameters:
        - filepath (str): Path to save the model (default: "recommender_knn_model.pkl").
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Train the model before saving.")

        with open(filepath, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {filepath} successfully.")

    def retrain_model(self, user_item_matrix: DataFrame) -> NearestNeighbors:
        """
        Retrain the k-NN model with an updated user-item matrix.

        Parameters:
        - user_item_matrix (DataFrame): Updated user-item interaction matrix.

        Returns:
        - NearestNeighbors: Retrained k-NN model.
        """
        print("Retraining k-NN model...")
        self.user_item_matrix = user_item_matrix
        return self.train_model()
        
    
