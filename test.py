import pickle
import pandas as pd
from utils import RecommenderUtils

# Paths to required files
movies_path = "movies.csv"
ratings_path = "ratings.csv"
knn_model_path = "recommender_knn_model.pkl"

# Load data and models
movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)

# Load the trained k-NN model
with open(knn_model_path, "rb") as model_file:
    knn_model = pickle.load(model_file)

# Create the user-item matrix dynamically from ratings
user_item_matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Instantiate RecommenderUtils
recommender_utils = RecommenderUtils(user_item_matrix, movies_df, knn_model=knn_model)

# Select a sample user to test recommendations
sample_user_id = ratings_df["userId"].sample(1).values[0]  # Randomly select a user

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
recommendations_knn = recommender_utils.get_recommendations_knn(sample_user_id)

# Display recommendations
if not recommendations_knn.empty:
    print(f"\nRecommendations for User {sample_user_id} (k-NN):\n")
    print(recommendations_knn)
else:
    print(f"\nNo recommendations could be generated for User {sample_user_id}.")
