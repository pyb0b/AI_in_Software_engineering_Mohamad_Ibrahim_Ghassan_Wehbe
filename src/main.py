from src.config.config import load_config
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model_knn import RecommenderTrainerKNN
from src.models.train_model_svd import RecommenderTrainerSVD
from src.utils.utils import RecommenderUtils
from src.models.evaluate_model import ModelEvaluator
from pandas import DataFrame


def main() -> None:
    """
    Main script to execute the workflow for loading data, preprocessing, 
    training models, evaluating them, and saving the results.
    """
    # Load configuration
    config = load_config()
    print("Loaded configuration:", config)

    # Paths to data from config
    movies_path = config.data.movies_path
    ratings_path = config.data.ratings_path

    # Load and preprocess data
    data_loader = DataLoader(movies_path, ratings_path)
    movies_df, ratings_df = data_loader.load_data()
    movies_df, ratings_df = data_loader.preprocess_data(movies_df, ratings_df)

    # Feature Engineering
    feature_engineer = FeatureEngineer(ratings_df)
    user_item_sparse, user_item_matrix = feature_engineer.create_user_item_matrix()

    # Initialize model trainers with parameters from config
    knn_trainer = RecommenderTrainerKNN(
        user_item_matrix,
        metric=config.knn_model.metric,
        algorithm=config.knn_model.algorithm,
    )
    svd_trainer = RecommenderTrainerSVD(
        user_item_matrix,
        num_factors=config.svd_model.num_factors,
    )

    # Train models
    knn_model = knn_trainer.train_model()
    svd_model = svd_trainer.train_model()

    # Save models
    knn_trainer.save_model(filepath="recommender_knn_model.pkl")
    svd_trainer.save_model(filepath="recommender_svd_model.pkl")

    # Initialize recommender utilities
    recommender_utils = RecommenderUtils(
        user_item_matrix, movies_df, knn_model=knn_model, svd_model=svd_model
    )

    # Evaluate models
    evaluator = ModelEvaluator(
        user_item_matrix, movies_df, svd_model=svd_model, knn_model=knn_model
    )

    # Split data into train and test sets for evaluation
    test_data = ratings_df.sample(frac=0.2, random_state=42)
    evaluator.evaluate_svd(test_data)

    # Test k-NN evaluation on a sample user
    sample_user_id = ratings_df["userId"].sample(1).values[0]
    user_test_data = test_data[test_data["userId"] == sample_user_id]

    if not user_test_data.empty:
        evaluator.evaluate_knn(user_test_data)
    else:
        print(f"No test data available for userId {sample_user_id}.")

    print("Main script executed successfully.")


if __name__ == "__main__":
    main()