import pandas as pd
from src.data.data_loader import DataLoader
from pandas import DataFrame

# Paths to your datasets
MOVIES_PATH = "movies.csv"
RATINGS_PATH = "ratings.csv"

# Initialize the DataLoader
data_loader = DataLoader(MOVIES_PATH, RATINGS_PATH)


def test_add_movie() -> None:
    """
    Test the `add_movie` function.

    Adds a new movie to the dataset and checks for successful addition.
    """
    new_movie = {
        "movieId": 10001,
        "title": "Test Movie",
        "genres": "Action|Adventure"
    }

    try:
        data_loader.add_movie(new_movie)
        print("Movie added successfully!")
    except Exception as e:
        print(f"Error adding movie: {e}")


def test_add_ratings() -> None:
    """
    Test the `add_ratings` function.

    Adds a set of new ratings to the dataset and checks for successful addition.
    """
    new_ratings: DataFrame = pd.DataFrame([
        {"userId": 999, "movieId": 1, "rating": 4.5},
        {"userId": 999, "movieId": 2, "rating": 3.0},
        {"userId": 999, "movieId": 3, "rating": 5.0}
    ])

    try:
        data_loader.add_ratings(new_ratings)
        print("Ratings added successfully!")
    except Exception as e:
        print(f"Error adding ratings: {e}")


if __name__ == "__main__":
    print("Testing add_movie function...")
    test_add_movie()

    print("\nTesting add_ratings function...")
    test_add_ratings()
