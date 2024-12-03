# evaluate_model.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Import RecommenderUtils from utils.py

from src.utils.utils import RecommenderUtils

class ModelEvaluator:


    def __init__(self, user_item_matrix, movies_df, svd_model=None, knn_model=None):
        self.user_item_matrix = user_item_matrix
        self.movies_df = movies_df
        self.svd_model = svd_model
        self.knn_model = knn_model


    def evaluate_svd(self, test_data):

        """
        Evaluate SVD model using RMSE and MAE metrics.
        Parameters:
        - test_data (DataFrame): Test dataset with userId, movieId, and actual ratings.
        Returns:
        - Dictionary with RMSE and MAE scores.
        """

        user_id_map = {uid: idx for idx, uid in enumerate(self.user_item_matrix.index)}
        movie_id_map = {mid: idx for idx, mid in enumerate(self.user_item_matrix.columns)}


        def predict(row):

            user_idx = user_id_map.get(row['userId'])
            movie_idx = movie_id_map.get(row['movieId'])

            if user_idx is not None and movie_idx is not None:
                return self.svd_model[user_idx, movie_idx]

            else:
                return np.nan


        test_data['predicted_rating'] = test_data.apply(predict, axis=1)
        valid_predictions = test_data.dropna(subset=['predicted_rating'])
        
        rmse = sqrt(mean_squared_error(valid_predictions['rating'], valid_predictions['predicted_rating']))
        mae = mean_absolute_error(valid_predictions['rating'], valid_predictions['predicted_rating'])

        print(f"SVD Model Evaluation: RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return {'RMSE': rmse, 'MAE': mae}


    def evaluate_knn(self, test_data, k=6):
        """
        Evaluate k-NN model using RMSE.
        Parameters:
        - test_data (DataFrame): Test dataset with userId, movieId, and actual ratings.
        - k (int): Number of neighbors to consider.
        Returns:
        - RMSE score.
        """
        predictions = []
        actuals = []

        for index, row in test_data.iterrows():
        
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']

            if movie_id in self.user_item_matrix.columns:

                user_index = self.user_item_matrix.index.get_loc(user_id)
                distances, indices = self.knn_model.kneighbors(
                    self.user_item_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=k
                )
                
                similar_users = indices.flatten()[1:]
                similar_user_ratings = self.user_item_matrix.iloc[similar_users, self.user_item_matrix.columns.get_loc(movie_id)]
                predicted_rating = (
                    similar_user_ratings[similar_user_ratings > 0].mean()
                    if not similar_user_ratings.empty
                    else np.nan
                )

                if not np.isnan(predicted_rating):
                
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)



        if predictions and actuals:
        
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            print(f"k-NN Model Evaluation: RMSE: {rmse:.4f}")
            return {'RMSE': rmse}
            
        else:
        
            print("No valid predictions could be made.")
            return {'RMSE': None}
            
        print(f"k-NN Model Evaluation: RMSE: {rmse:.4f}")
        return {'RMSE': rmse}

