import pytest
import pandas as pd
import numpy as np
from src.models.evaluate_model import ModelEvaluator


@pytest.fixture
def user_item_matrix():
    data = {
        "movieId": [1, 2, 3],
        "userId": [1, 1, 2],
        "rating": [5.0, 3.0, 4.0]
    }
    df = pd.DataFrame(data)
    return df.pivot(index="userId", columns="movieId", values="rating").fillna(0)


@pytest.fixture
def movies_df():
    return pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": ["Action", "Comedy", "Drama"]
    })


@pytest.fixture
def svd_model(user_item_matrix):
    """
    Mock an SVD model for testing.
    This is a simplified version where we just return the user-item matrix.
    """
    user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
    user_item_matrix_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = np.linalg.svd(user_item_matrix_demeaned, full_matrices=False)
    sigma = np.diag(sigma)

    return np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)


@pytest.fixture
def evaluator(user_item_matrix, movies_df, svd_model):
    return ModelEvaluator(user_item_matrix, movies_df, svd_model=svd_model)


def test_evaluate_svd(evaluator):
    sample_data = pd.DataFrame({
        "userId": [1, 2],
        "movieId": [1, 3],
        "rating": [5.0, 4.0]
    })

    # Call evaluate_svd and check results
    result = evaluator.evaluate_svd(sample_data)
    assert "RMSE" in result
    assert "MAE" in result
    assert result["RMSE"] is not None
    assert result["MAE"] is not None
