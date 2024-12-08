import pickle
import pandas as pd
from pandas import DataFrame
from src.utils.utils import RecommenderUtils


# Paths to required files
MOVIES_PATH = "movies.csv"
RATINGS_PATH = "ratings.csv"
KNN_MODEL_PATH = "recommender_knn_model.pkl"


def main() -> None:
    """
    Test script to validate the k-NN recommender system's functionality.

    Loads data, models, and a sample user's information to generate recommendations.
    """
    # Load data
    movies_df: DataFrame = pd.read_csv(MOVIES_PATH)
    ratings_df: DataFrame = pd.read_csv(RATINGS_PATH)

    # Load the trained k-NN model
    with open(KNN_MODEL_PATH, "rb") as model_file:
        knn_model = pickle.load(model_file)

    # Create the user-item matrix dynamically from ratings
    user_item_matrix: DataFrame = ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    # Instantiate RecommenderUtils
    recommender_utils = RecommenderUtils(user_item_matrix, movies_df, knn_model=knn_model)

    # Select a sample user to test recommendations
    sample_user_id: int = ratings_df["userId"].sample(1).values[0]  # Randomly select a user

    # Get movies the user has already liked and rated
    user_ratings = ratings_df[(ratings_df["userId"] == sample_user_id) & (ratings_df["rating"] > 4)]
    liked_movie_titles = movies_df[movies_df["movieId"].isin(user_ratings["movieId"])]

    # Display liked and rated movies
    if not liked_movie_titles.empty:
        print(f"Movies already liked and rated by User {sample_user_id}:\n")
        print(liked_movie_titles[["title"]])
    else:
        print(f"User {sample_user_id} has not liked or rated any movies above the threshold.")

    # Generate recommendations
    recommendations_knn: DataFrame = recommender_utils.get_recommendations_knn(sample_user_id)

    # Display recommendations
    if not recommendations_knn.empty:
        print(f"\nRecommendations for User {sample_user_id} (k-NN):\n")
        print(recommendations_knn)
    else:
        print(f"\nNo recommendations could be generated for User {sample_user_id}.")


if __name__ == "__main__":
    main()
