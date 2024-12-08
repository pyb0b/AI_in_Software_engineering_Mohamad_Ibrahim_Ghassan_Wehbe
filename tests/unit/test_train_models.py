import pytest
from src.models.train_model_knn import RecommenderTrainerKNN
from src.models.train_model_svd import RecommenderTrainerSVD
import pandas as pd

@pytest.fixture
def user_item_matrix():
    data = {
        "movieId": [1, 2, 3],
        "userId": [1, 1, 2],
        "rating": [5.0, 3.0, 4.0]
    }
    df = pd.DataFrame(data)
    return df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

def test_train_knn(user_item_matrix):
    trainer = RecommenderTrainerKNN(user_item_matrix)
    model = trainer.train_model()
    assert model is not None

def test_train_svd(user_item_matrix):
    trainer = RecommenderTrainerSVD(user_item_matrix, num_factors=1)  # Set num_factors <= min(matrix dimensions)
    model = trainer.train_model()
    assert model is not None

