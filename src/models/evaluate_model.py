import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from pandas import DataFrame
from typing import Optional, Dict
import mlflow

# Import RecommenderUtils from utils.py
from src.utils.utils import RecommenderUtils


class ModelEvaluator:
    """
    A class for evaluating collaborative filtering models like SVD and k-NN.
    """

    def __init__(
        self,
        user_item_matrix: DataFrame,
        movies_df: DataFrame,
        svd_model: Optional[np.ndarray] = None,
        knn_model: Optional[object] = None,
    ) -> None:
        """
        Initialize the ModelEvaluator.

        Parameters:
        - user_item_matrix (DataFrame): User-item interaction matrix.
        - movies_df (DataFrame): DataFrame containing movie information.
        - svd_model (Optional[np.ndarray]): Pre-trained SVD model.
        - knn_model (Optional[object]): Pre-trained k-NN model.
        """
        self.user_item_matrix = user_item_matrix
        self.movies_df = movies_df
        self.svd_model = svd_model
        self.knn_model = knn_model

    def evaluate_svd(self, test_data: DataFrame) -> Dict[str, float]:
        """
        Evaluate the SVD model using RMSE and MAE metrics.

        Parameters:
        - test_data (DataFrame): Test dataset with 'userId', 'movieId', and actual ratings.

        Returns:
        - Dictionary with RMSE and MAE scores.
        """
        user_id_map = {uid: idx for idx, uid in enumerate(self.user_item_matrix.index)}
        movie_id_map = {mid: idx for idx, mid in enumerate(self.user_item_matrix.columns)}

        def predict(row: pd.Series) -> Optional[float]:
            user_idx = user_id_map.get(row["userId"])
            movie_idx = movie_id_map.get(row["movieId"])

            if user_idx is not None and movie_idx is not None:
                return self.svd_model[user_idx, movie_idx]
            return np.nan
        
        with mlflow.start_run(run_name="SVD Model Evaluation"):
            test_data["predicted_rating"] = test_data.apply(predict, axis=1)
            valid_predictions = test_data.dropna(subset=["predicted_rating"])
            rmse = sqrt(mean_squared_error(valid_predictions["rating"], valid_predictions["predicted_rating"]))
            mae = mean_absolute_error(valid_predictions["rating"], valid_predictions["predicted_rating"])

            # Log metrics
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)

            print(f"SVD Model Evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}")
            return {"RMSE": rmse, "MAE": mae}

    def evaluate_knn(self, test_data: DataFrame, k: int = 6) -> Dict[str, Optional[float]]:
        """
        Evaluate the k-NN model using RMSE.

        Parameters:
        - test_data (DataFrame): Test dataset with 'userId', 'movieId', and actual ratings.
        - k (int): Number of neighbors to consider.

        Returns:
        - Dictionary containing the RMSE score.
        """
        predictions = []
        actuals = []

        for _, row in test_data.iterrows():
            user_id = row["userId"]
            movie_id = row["movieId"]
            actual_rating = row["rating"]

            if movie_id in self.user_item_matrix.columns:
                user_index = self.user_item_matrix.index.get_loc(user_id)
                distances, indices = self.knn_model.kneighbors(
                    self.user_item_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=k
                )

                similar_users = indices.flatten()[1:]
                similar_user_ratings = self.user_item_matrix.iloc[
                    similar_users, self.user_item_matrix.columns.get_loc(movie_id)
                ]
                predicted_rating = (
                    similar_user_ratings[similar_user_ratings > 0].mean()
                    if not similar_user_ratings.empty
                    else np.nan
                )

                if not np.isnan(predicted_rating):
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)

        if predictions and actuals:
            rmse = sqrt(mean_squared_error(actuals, predictions))
            print(f"k-NN Model Evaluation: RMSE: {rmse:.4f}")
            return {"RMSE": rmse}

        print("No valid predictions could be made.")
        return {"RMSE": None}
