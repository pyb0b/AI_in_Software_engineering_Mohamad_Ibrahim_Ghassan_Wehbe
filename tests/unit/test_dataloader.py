import pytest
import pandas as pd
from src.data.data_loader import DataLoader

# Paths to sample datasets
MOVIES_PATH = "movies.csv"
RATINGS_PATH = "ratings.csv"

@pytest.fixture
def dataloader():
    return DataLoader(MOVIES_PATH, RATINGS_PATH)

def test_load_data(dataloader):
    movies_df, ratings_df = dataloader.load_data()
    assert isinstance(movies_df, pd.DataFrame)
    #assert isinstance(ratings_df, pd.DataFrame)
    assert not movies_df.empty
    assert not ratings_df.empty

def test_preprocess_data(dataloader):
    movies_df, ratings_df = dataloader.load_data()
    movies_df, ratings_df = dataloader.preprocess_data(movies_df, ratings_df)
    assert "title" in movies_df.columns
    assert "rating" in ratings_df.columns
