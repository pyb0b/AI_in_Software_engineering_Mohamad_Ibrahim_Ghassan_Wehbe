
# feature_engineering.py

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df

    def create_user_item_matrix(self):
        # Create a pivot table for user-item interactions
        user_item_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values='rating')
        
        # Fill NaN with 0 for collaborative filtering
        user_item_matrix = user_item_matrix.fillna(0)
        
        # Convert to a sparse matrix for memory efficiency
        user_item_sparse = csr_matrix(user_item_matrix.values)
        
        print("User-item matrix created successfully.")
        return user_item_sparse, user_item_matrix
        
    def update_user_item_matrix(self, ratings_df):
        """
        Update the user-item matrix with new ratings data.

        Parameters:
        - ratings_df: Updated ratings DataFrame

        Returns:
        - Sparse matrix for memory efficiency
        - Full user-item matrix as DataFrame for compatibility
        """
        self.ratings_df = ratings_df
        return self.create_user_item_matrix()