# Special Recommender System

This project implements a movie recommender system using collaborative filtering. It predicts user preferences based on their own movie ratings and ratings provided by others. The system is designed with MLOps principles, ensuring efficient deployment, monitoring, and experiment tracking.

---

## Features
- **Collaborative Filtering**: Recommend movies based on user ratings using KNN and SVD models.
- **API Integration**: FastAPI-powered endpoints for training and making predictions.
- **MLOps Practices**: Includes logging, monitoring, and experiment tracking with Prometheus and MLflow.
- **Interactive GUI**: A simple GUI for testing the recommender system.

---

## Dependencies
The project requires the following dependencies:
- `numpy`
- `pandas`
- `scikit-learn`
- `fastapi`
- `prometheus-fastapi-instrumentator`
- `requests`

Install these dependencies using Poetry:
```bash
poetry install

---

## Usage

### **1. Running the Project**
1. Navigate to the project directory:
   ```bash
   cd <project-directory>
2. Activate the Poetry environment:
   poetry shell
3. Start the FastAPI application:
   uvicorn src.app:app --reload
4. Open your browser and navigate to:
   Root Endpoint: http://127.0.0.1:8000

---

### **What’s Included**
- **Project Overview**: Brief description of purpose and features.
- **Dependencies**: All required libraries and installation instructions.
- **Usage Instructions**: Step-by-step commands for running the project.
- **API Details**: Descriptions, expected input/output, and example usage for endpoints.
- **Docker Commands**: For building and running the container.
- **MLOps Integration**: Details about MLflow and Prometheus monitoring.
- **Future Enhancements**: Suggestions for improving the project.

---

## API Endpoints

This section provides details about the available API endpoints, their paths, expected input formats, and example responses.

---

### **1. Train a Model**
- **Method**: `POST`
- **Endpoint Path**: `/train`
- **Description**: Train a collaborative filtering model (e.g., KNN or SVD) with specified hyperparameters.

#### **Expected Input Format (JSON)**:
```json
{
  "model_type": "knn",  // Model type: "knn" or "svd"
  "hyperparameters": {  // Hyperparameters for the model
    "k": 10            // Example hyperparameter for KNN
  }
}

---

### **2. Get Recommendations**
- **Method**: `POST`
- **Endpoint Path**: `/recommend`
- **Description**: Generate movie recommendations for a specific user based on their ratings and collaborative filtering.
#### **Expected Input Format (JSON)**:
```json
{
  "user_id": 123  // Unique ID of the user requesting recommendations
}

---

### **3. Monitor Metrics**
- **Method**: `GET`
- **Endpoint Path**: `/metrics`
- **Description**: Retrieve metrics about the API’s performance, usage, and other monitoring data tracked by Prometheus.
