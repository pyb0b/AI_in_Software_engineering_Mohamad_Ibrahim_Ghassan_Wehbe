import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import pickle
from pandas import DataFrame
from typing import Optional
import mlflow


class RecommenderTrainerSVD:
    """
    A class to train and manage a collaborative filtering model using Singular Value Decomposition (SVD).
    """

    def __init__(self, user_item_matrix: DataFrame, num_factors: int = 50) -> None:
        """
        Initialize the SVD recommender trainer.

        Parameters:
        - user_item_matrix (DataFrame): User-item interaction matrix.
        - num_factors (int): Number of latent factors to retain during SVD (default: 50).
        """
        self.user_item_matrix = user_item_matrix
        self.num_factors = num_factors
        # self.num_factors = min(num_factors, min(user_item_matrix.shape))  # Adjust num_factors to fit matrix dimensions
        self.model: Optional[np.ndarray] = None

    def train_model(self) -> np.ndarray:
        """
        Train a collaborative filtering model using Singular Value Decomposition (SVD).

        Returns:
        - np.ndarray: Reconstructed user-item matrix with predicted ratings.
        """
        # Compute the mean user ratings
        user_ratings_mean = np.mean(self.user_item_matrix.values, axis=1)
        user_item_matrix_demeaned = self.user_item_matrix.values - user_ratings_mean.reshape(-1, 1)
        
        with mlflow.start_run(run_name="SVD Model Training"):
            # Log parameters
            mlflow.log_param("num_factors", self.num_factors)
            mlflow.log_param("matrix_shape", self.user_item_matrix.shape)

            # Train the SVD model
            U, sigma, Vt = np.linalg.svd(user_item_matrix_demeaned, full_matrices=False)
            sigma = np.diag(sigma)
            self.model = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

            # Log the model as an artifact
            model_path = "recommender_svd_model.pkl"
            with open(model_path, "wb") as file:
                pickle.dump(self.model, file)
            mlflow.log_artifact(model_path)

            print("SVD model training completed successfully.")
            return self.model

    def save_model(self, filepath: str = "recommender_svd_model.pkl") -> None:
        """
        Save the trained SVD model to a file.

        Parameters:
        - filepath (str): Path to save the model (default: "recommender_svd_model.pkl").
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Train the model before saving.")

        with open(filepath, "wb") as file:
            pickle.dump(self.model, file)

        print(f"SVD model saved to {filepath} successfully.")

    def retrain_model(self, user_item_matrix: DataFrame) -> np.ndarray:
        """
        Retrain the SVD model with an updated user-item matrix.

        Parameters:
        - user_item_matrix (DataFrame): Updated user-item interaction matrix.

        Returns:
        - np.ndarray: Reconstructed user-item matrix with predicted ratings.
        """
        print("Retraining SVD model...")
        self.user_item_matrix = user_item_matrix
        return self.train_model()
