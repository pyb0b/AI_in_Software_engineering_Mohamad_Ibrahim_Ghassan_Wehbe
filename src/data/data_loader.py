
# data_loader.py

import pandas as pd

class DataLoader:
    def __init__(self, movies_path, ratings_path):
        self.movies_path = movies_path
        self.ratings_path = ratings_path

    def load_data(self):
        try:
            # Load movies data
            movies_df = pd.read_csv(self.movies_path)
            # Load ratings data
            ratings_df = pd.read_csv(self.ratings_path)
            print("Data loaded successfully.")
            return movies_df, ratings_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def preprocess_data(self, movies_df, ratings_df):
        # Drop timestamp column as per requirements
        ratings_df = ratings_df.drop(columns=['timestamp'])
        
        # Handle missing data if necessary
        movies_df = movies_df.dropna(subset=['title', 'genres'])
        ratings_df = ratings_df.dropna(subset=['userId', 'movieId', 'rating'])
        
        # Additional preprocessing can be added here if needed
        print("Data preprocessed successfully.")
        return movies_df, ratings_df

    def add_movie(self, movie):
        """
        Add a new movie to the movies dataset.

        Parameters:
        - movie (dict): A dictionary with keys 'movieId', 'title', and 'genres'.
        """
        movies_df = pd.read_csv(self.movies_path)

        # Check if the movieId already exists
        if movie['movieId'] in movies_df['movieId'].values:
            print(f"Movie with ID {movie['movieId']} already exists. Skipping addition.")
            return

        # Add the new movie
        new_movie = pd.DataFrame([movie])
        updated_movies = pd.concat([movies_df, new_movie], ignore_index=True)

        # Save back to CSV
        updated_movies.to_csv(self.movies_path, index=False)
        print(f"Movie '{movie['title']}' added successfully!")

    def add_ratings(self, new_ratings):
        """
        Add new ratings to the ratings dataset.

        Parameters:
        - new_ratings (DataFrame): A DataFrame with columns 'userId', 'movieId', and 'rating'.
        """
        ratings_df = pd.read_csv(self.ratings_path)
        
        new_ratings['timestamp'] = 0  # Default value for timestamp

        # Combine and drop duplicates
        updated_ratings = pd.concat([ratings_df, new_ratings], ignore_index=True)
        updated_ratings = updated_ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')

        # Save back to CSV
        updated_ratings.to_csv(self.ratings_path, index=False)
        print(f"New ratings added successfully!")