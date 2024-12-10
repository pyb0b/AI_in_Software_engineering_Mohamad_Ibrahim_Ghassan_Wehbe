from pydantic import BaseModel, validator
from omegaconf import OmegaConf
import os

# Pydantic models for validation
class DataConfig(BaseModel):
    movies_path: str
    ratings_path: str

    @validator("movies_path", "ratings_path")
    def validate_paths(cls, path):
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        return path


class KNNModelConfig(BaseModel):
    metric: str
    algorithm: str
    n_neighbors: int

    @validator("metric")
    def validate_metric(cls, value):
        if value not in {"cosine", "euclidean"}:
            raise ValueError("Metric must be 'cosine' or 'euclidean'")
        return value

    @validator("algorithm")
    def validate_algorithm(cls, value):
        if value not in {"brute", "kd_tree"}:
            raise ValueError("Algorithm must be 'brute' or 'kd_tree'")
        return value


class SVDModelConfig(BaseModel):
    num_factors: int

    @validator("num_factors")
    def validate_num_factors(cls, value):
        if value <= 0:
            raise ValueError("Number of factors must be greater than 0")
        return value


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str


class Config(BaseModel):
    data: DataConfig
    knn_model: KNNModelConfig
    svd_model: SVDModelConfig
    mlflow: MLflowConfig


# Function to load and validate the configuration
def load_config(config_path: str = "src/config/config.yaml") -> Config:
    """
    Load and validate the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Config: Validated configuration object.
    """
    raw_config = OmegaConf.load(config_path)  # Load YAML using OmegaConf
    config_dict = OmegaConf.to_container(raw_config, resolve=True)  # Convert to dictionary
    return Config(**config_dict)  # Validate using Pydantic
