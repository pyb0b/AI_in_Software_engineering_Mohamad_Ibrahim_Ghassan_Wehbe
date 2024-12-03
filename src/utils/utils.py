
# utils.py

import numpy as np
import pandas as pd

class RecommenderUtils:
    def __init__(self, user_item_matrix, movies_df, knn_model=None, svd_model=None):
        self.user_item_matrix = user_item_matrix
        self.movies_df = movies_df
        self.knn_model = knn_model
        self.svd_model = svd_model

    def get_recommendations_knn(self, user_id, n_neighbors=6):
        """
        Generate movie recommendations for a user based on similar users using the k-NN model.

        Parameters:
        - user_id (int): The ID of the user for whom to generate recommendations
        - n_neighbors (int): Number of neighbors (similar users) to consider

        Returns:
        - DataFrame with recommended movie titles and average rating
        """
        if user_id not in self.user_item_matrix.index:
            print("User not found in the user-item matrix.")
            return pd.DataFrame()

        user_index = self.user_item_matrix.index.get_loc(user_id)
        distances, indices = self.knn_model.kneighbors(self.user_item_matrix.iloc[user_index, :].values.reshape(1, -1), 
                                                       n_neighbors=n_neighbors)
        
        similar_users = indices.flatten()[1:]
        user_ratings = self.user_item_matrix.loc[user_id]
        recommendations = pd.Series(dtype='float64')

        for user in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[user]
            unrated_movies = similar_user_ratings[similar_user_ratings > 4].index.difference(user_ratings[user_ratings > 0].index)
            recommendations = pd.concat([recommendations, similar_user_ratings[unrated_movies]])

        recommendations = recommendations.groupby(recommendations.index).mean()
        recommended_movies = recommendations.sort_values(ascending=False).head(10)

        recommended_movie_titles = self.movies_df[self.movies_df['movieId'].isin(recommended_movies.index)].set_index('movieId')
        recommended_movie_titles['average_rating'] = recommended_movies

        print(f"Recommended Movies for User {user_id} (k-NN):", recommended_movie_titles)
        return recommended_movie_titles

    def get_recommendations_svd(self, user_id, n_recommendations=10):
        """
        Generate movie recommendations for a user based on the SVD model.

        Parameters:
        - user_id (int): The ID of the user for whom to generate recommendations
        - n_recommendations (int): Number of recommendations to return

        Returns:
        - DataFrame with recommended movie titles and predicted rating
        """
        if user_id not in self.user_item_matrix.index:
            print("User not found in the user-item matrix.")
            return pd.DataFrame()

        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_predictions = self.svd_model[user_index, :].flatten()

        recommendations = np.argsort(-user_predictions)[:n_recommendations]
        recommended_movie_ids = self.user_item_matrix.columns[recommendations].tolist()

        recommended_movie_titles = self.movies_df[self.movies_df['movieId'].isin(recommended_movie_ids)].set_index('movieId')
        recommended_movie_titles['predicted_rating'] = [user_predictions[self.user_item_matrix.columns.get_loc(mid)] for mid in recommended_movie_ids]
        
        print(f"Recommended Movies for User {user_id} (SVD):", recommended_movie_titles)
        return recommended_movie_titles
