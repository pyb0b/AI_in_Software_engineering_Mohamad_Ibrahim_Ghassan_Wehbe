import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from typing import Tuple


class FeatureEngineer:
    """
    A class for engineering features from the ratings dataset.
    Specifically focuses on creating and updating user-item interaction matrices.
    """

    def __init__(self, ratings_df: DataFrame) -> None:
        """
        Initialize the FeatureEngineer with a ratings DataFrame.

        Parameters:
        - ratings_df (DataFrame): The ratings dataset containing 'userId', 'movieId', and 'rating' columns.
        """
        self.ratings_df = ratings_df

    def create_user_item_matrix(self) -> Tuple[csr_matrix, DataFrame]:
        """
        Create a user-item interaction matrix for collaborative filtering.

        Returns:
        - user_item_sparse (csr_matrix): Sparse matrix of user-item interactions.
        - user_item_matrix (DataFrame): Full user-item matrix with NaN values replaced by 0.
        """
        # Create a pivot table for user-item interactions
        user_item_matrix = self.ratings_df.pivot(index="userId", columns="movieId", values="rating")

        # Fill NaN with 0 for collaborative filtering
        user_item_matrix = user_item_matrix.fillna(0)

        # Convert to a sparse matrix for memory efficiency
        user_item_sparse = csr_matrix(user_item_matrix.values)

        print("User-item matrix created successfully.")
        return user_item_sparse, user_item_matrix

    def update_user_item_matrix(self, ratings_df: DataFrame) -> Tuple[csr_matrix, DataFrame]:
        """
        Update the user-item matrix with new ratings data.

        Parameters:
        - ratings_df (DataFrame): Updated ratings DataFrame containing 'userId', 'movieId', and 'rating' columns.

        Returns:
        - user_item_sparse (csr_matrix): Sparse matrix of updated user-item interactions.
        - user_item_matrix (DataFrame): Full updated user-item matrix with NaN values replaced by 0.
        """
        self.ratings_df = ratings_df
        return self.create_user_item_matrix()
