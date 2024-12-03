from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from contextlib import asynccontextmanager

# Import components
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from train_model_knn import RecommenderTrainerKNN
from train_model_svd import RecommenderTrainerSVD
from utils import RecommenderUtils
from evaluate_model import ModelEvaluator
from monitoring import setup_monitoring

# Initialize FastAPI with a lifespan manager
app = FastAPI()

# Initialize monitoring
setup_monitoring(app)

@app.get("/")
async def root():
    return {"message": "API is running and monitoring is enabled!"}

# Global variables for model and data
movies_df = None
ratings_df = None
user_item_sparse = None
user_item_matrix = None
user_item_matrix = None
knn_model = None
svd_model = None
recommender_utils = None

# Paths to data
MOVIES_PATH = 'movies.csv'
RATINGS_PATH = 'ratings.csv'


class RecommendationRequest(BaseModel):
    user_id: int
    model_type: str  # 'knn' or 'svd'

    class Config:
        # Disable protected namespace warnings
        protected_namespaces = ()


class Movie(BaseModel):
    movieId: int
    title: str
    genres: str


class Rating(BaseModel):
    userId: int
    movieId: int
    rating: float


@app.on_event("startup")
async def load_data_and_initialize():
    """
    Load data, initialize models, and prepare global variables during startup.
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

    # Initialize recommender utils
    recommender_utils = RecommenderUtils(user_item_matrix, movies_df, knn_model=knn_model, svd_model=svd_model)

    print("Application initialized successfully!")


@app.post("/add_movie/")
async def add_movie(movie: Movie):
    """
    Add a new movie to the dataset.
    """
    global movies_df

    data_loader = DataLoader(MOVIES_PATH, RATINGS_PATH)
    try:
        # Add the movie to the dataset
        data_loader.add_movie(movie.dict())
        movies_df = pd.read_csv(MOVIES_PATH)
        return {"message": f"Movie '{movie.title}' added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/add_ratings/")
async def add_ratings(ratings: List[Rating]):
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
async def get_recommendations(request: RecommendationRequest):
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
async def evaluate_svd():
    # Sample test data for evaluation
    test_data = pd.read_csv(RATINGS_PATH).sample(frac=0.2, random_state=42)
    evaluator = ModelEvaluator(user_item_matrix, movies_df, svd_model=svd_model)

    evaluation_results = evaluator.evaluate_svd(test_data)
    return {"evaluation_results": evaluation_results}

@app.get("/evaluate_knn/{user_id}")
async def evaluate_knn(user_id: int):
    evaluator = ModelEvaluator(user_item_matrix, movies_df, knn_model=knn_model)

    evaluation_results = evaluator.evaluate_knn(user_id)
    return {"user_id": user_id, "evaluation_results": evaluation_results}

