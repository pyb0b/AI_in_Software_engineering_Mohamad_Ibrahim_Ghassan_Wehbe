from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

# Import components
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model_knn import RecommenderTrainerKNN
from src.models.train_model_svd import RecommenderTrainerSVD
from src.utils.utils import RecommenderUtils
from src.models.evaluate_model import ModelEvaluator
from src.monitoring.monitoring import setup_monitoring

# Initialize FastAPI with monitoring
app = FastAPI()
setup_monitoring(app)

# Global variables for models and data
movies_df: pd.DataFrame = None
ratings_df: pd.DataFrame = None
user_item_sparse = None
user_item_matrix: pd.DataFrame = None
knn_model = None
svd_model = None
recommender_utils: RecommenderUtils = None

# Paths to data
MOVIES_PATH = "movies.csv"
RATINGS_PATH = "ratings.csv"


class RecommendationRequest(BaseModel):
    """
    Request schema for generating recommendations.
    """
    user_id: int
    model_type: str  # 'knn' or 'svd'


class Movie(BaseModel):
    """
    Schema for adding a new movie to the dataset.
    """
    movieId: int
    title: str
    genres: str


class Rating(BaseModel):
    """
    Schema for adding a new rating to the dataset.
    """
    userId: int
    movieId: int
    rating: float


@app.on_event("startup")
async def load_data_and_initialize() -> None:
    """
    Load data, initialize models, 
    and prepare global variables during app startup.
    """
    global movies_df, ratings_df, user_item_sparse, user_item_matrix, knn_model, svd_model, recommender_utils

    # Load and preprocess data
    data_loader = DataLoader(MOVIES_PATH, RATINGS_PATH)
    movies_df, ratings_df = data_loader.load_data()
    movies_df, ratings_df = data_loader.preprocess_data(movies_df, ratings_df)

    # Feature engineering
    feature_engineer = FeatureEngineer(ratings_df)
    user_item_sparse, user_item_matrix = feature_engineer.create_user_item_matrix()

    # Train models
    knn_trainer = RecommenderTrainerKNN(user_item_matrix)
    svd_trainer = RecommenderTrainerSVD(user_item_matrix)
    knn_model = knn_trainer.train_model()
    svd_model = svd_trainer.train_model()

    # Initialize recommender utilities
    recommender_utils = RecommenderUtils(user_item_matrix, movies_df, knn_model=knn_model, svd_model=svd_model)

    print("Application initialized successfully!")


@app.get("/")
async def root() -> dict:
    """
    Root endpoint to confirm API is running.
    """
    return {"message": "API is running and monitoring is enabled!"}


@app.post("/add_movie/")
async def add_movie(movie: Movie) -> dict:
    """
    Add a new movie to the dataset.
    """
    global movies_df
    data_loader = DataLoader(MOVIES_PATH, RATINGS_PATH)

    try:
        data_loader.add_movie(movie.dict())
        movies_df = pd.read_csv(MOVIES_PATH)
        return {"message": f"Movie '{movie.title}' added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/add_ratings/")
async def add_ratings(ratings: List[Rating]) -> dict:
    """
    Add new ratings to the dataset and update models.
    """
    global ratings_df, user_item_sparse, user_item_matrix, knn_model, svd_model, recommender_utils

    try:
        # Convert ratings to DataFrame
        new_ratings = pd.DataFrame([r.dict() for r in ratings])

        # Add ratings to the dataset
        data_loader = DataLoader(MOVIES_PATH, RATINGS_PATH)
        data_loader.add_ratings(new_ratings)
        ratings_df = pd.read_csv(RATINGS_PATH)

        # Update user-item matrix
        feature_engineer = FeatureEngineer(ratings_df)
        user_item_sparse, user_item_matrix = feature_engineer.update_user_item_matrix(ratings_df)

        # Retrain models
        knn_trainer = RecommenderTrainerKNN(user_item_matrix)
        svd_trainer = RecommenderTrainerSVD(user_item_matrix)
        knn_model = knn_trainer.retrain_model(user_item_matrix)
        svd_model = svd_trainer.retrain_model(user_item_matrix)

        # Update recommender utils
        recommender_utils = RecommenderUtils(user_item_matrix, movies_df, knn_model=knn_model, svd_model=svd_model)

        return {"message": "New ratings added, user-item matrix updated, and models retrained successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend/")
async def get_recommendations(request: RecommendationRequest) -> dict:
    """
    Generate movie recommendations for a user.
    """
    user_id = request.user_id
    model_type = request.model_type.lower()

    if model_type not in ["knn", "svd"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'knn' or 'svd'.")

    if model_type == "knn":
        recommendations = recommender_utils.get_recommendations_knn(user_id)
    else:
        recommendations = recommender_utils.get_recommendations_svd(user_id)

    return {"user_id": user_id, "recommendations": recommendations.to_dict()}


@app.get("/evaluate_svd/")
async def evaluate_svd() -> dict:
    """
    Evaluate the SVD model using RMSE and MAE metrics.
    """
    test_data = pd.read_csv(RATINGS_PATH).sample(frac=0.2, random_state=42)
    evaluator = ModelEvaluator(user_item_matrix, movies_df, svd_model=svd_model)

    evaluation_results = evaluator.evaluate_svd(test_data)
    return {"evaluation_results": evaluation_results}


@app.get("/evaluate_knn/{user_id}")
async def evaluate_knn(user_id: int) -> dict:
    """
    Evaluate the k-NN model for a specific user.
    """
    evaluator = ModelEvaluator(user_item_matrix, movies_df, knn_model=knn_model)

    evaluation_results = evaluator.evaluate_knn(user_id)
    return {"user_id": user_id, "evaluation_results": evaluation_results}
