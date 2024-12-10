import pandas as pd
from pandas import DataFrame
from typing import Tuple, Optional, Dict


class DataLoader:
    """
    A class to handle loading, preprocessing, and managing movie and ratings datasets.
    """

    def __init__(self, movies_path: str, ratings_path: str) -> None:
        """
        Initialize the DataLoader with paths to the datasets.

        Parameters:
        - movies_path (str): Path to the movies CSV file.
        - ratings_path (str): Path to the ratings CSV file.
        """
        self.movies_path = movies_path
        self.ratings_path = ratings_path

    def load_data(self) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """
        Load the movies and ratings datasets.

        Returns:
        - Tuple containing:
            - movies_df (Optional[DataFrame]): The loaded movies DataFrame, or None if an error occurs.
            - ratings_df (Optional[DataFrame]): The loaded ratings DataFrame, or None if an error occurs.
        """
        try:
            movies_df = pd.read_csv(self.movies_path)  # Load movies data
            ratings_df = pd.read_csv(self.ratings_path)  # Load ratings data
            print("Data loaded successfully.")
            return movies_df, ratings_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def preprocess_data(
        self, movies_df: DataFrame, ratings_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Preprocess the movies and ratings datasets.

        Parameters:
        - movies_df (DataFrame): The movies dataset.
        - ratings_df (DataFrame): The ratings dataset.

        Returns:
        - Tuple containing:
            - movies_df (DataFrame): The preprocessed movies DataFrame.
            - ratings_df (DataFrame): The preprocessed ratings DataFrame.
        """
        # Drop timestamp column as per requirements
        ratings_df = ratings_df.drop(columns=["timestamp"])

        # Handle missing data
        movies_df = movies_df.dropna(subset=["title", "genres"])
        ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating"])

        print("Data preprocessed successfully.")
        return movies_df, ratings_df

    def add_movie(self, movie: Dict[str, str]) -> None:
        """
        Add a new movie to the movies dataset.

        Parameters:
        - movie (dict): A dictionary with keys 'movieId', 'title', and 'genres'.
        """
        movies_df = pd.read_csv(self.movies_path)

        # Check if the movieId already exists
        if int(movie["movieId"]) in movies_df["movieId"].values:
            print(f"Movie with ID {movie['movieId']} already exists. Skipping addition.")
            return

        # Add the new movie
        new_movie = pd.DataFrame([movie])
        updated_movies = pd.concat([movies_df, new_movie], ignore_index=True)

        # Save back to CSV
        updated_movies.to_csv(self.movies_path, index=False)
        print(f"Movie '{movie['title']}' added successfully!")

    def add_ratings(self, new_ratings: DataFrame) -> None:
        """
        Add new ratings to the ratings dataset.

        Parameters:
        - new_ratings (DataFrame): A DataFrame with columns 'userId', 'movieId', and 'rating'.
        """
        ratings_df = pd.read_csv(self.ratings_path)

        new_ratings["timestamp"] = 0  # Default value for timestamp

        # Combine and drop duplicates
        updated_ratings = pd.concat([ratings_df, new_ratings], ignore_index=True)
        updated_ratings = updated_ratings.drop_duplicates(
            subset=["userId", "movieId"], keep="last"
        )

        # Save back to CSV
        updated_ratings.to_csv(self.ratings_path, index=False)
        print("New ratings added successfully!")
