import tkinter as tk
from tkinter import messagebox
import requests
from tkinter import font

# Functions to handle API calls
def recommend():
    try:
        user_id = recommend_user_id.get()
        model_type = model_choice.get()
        if not user_id or not model_type:
            raise ValueError("User ID and Model Type are required.")
        
        payload = {"user_id": int(user_id), "model_type": model_type}
        response = requests.post("http://127.0.0.1:8000/recommend/", json=payload)
        #recommend_result_label.config(text=f"Response: {response.json()}")
        data = response.json()
        
        # Extracting movie titles
        if "recommendations" in data and "title" in data["recommendations"]:
            titles = data["recommendations"]["title"].values()
            formatted_titles = "\n".join(titles)
            recommend_result_label.config(text=f"Recommended Movies:\n{formatted_titles}")
        else:
            recommend_result_label.config(text="No recommendations found.")
            
    except Exception as e:
        recommend_result_label.config(text=f"Error: {e}")

def add_movie():
    try:
        movie_id = movie_id_entry.get()
        title = movie_title_entry.get()
        genres = movie_genres_entry.get()
        if not movie_id or not title or not genres:
            raise ValueError("All fields are required.")
        
        payload = {"movieId": int(movie_id), "title": title, "genres": genres}
        response = requests.post("http://127.0.0.1:8000/add_movie/", json=payload)
        #add_movie_result_label.config(text=f"Response: {response.json()}")
        data = response.json()
        
        # Displaying the success message
        if "message" in data:
            add_movie_result_label.config(text=data["message"])
        else:
            add_movie_result_label.config(text="Failed to add movie. No message received.")
    except Exception as e:
        add_movie_result_label.config(text=f"Error: {e}")

def add_rating():
    try:
        user_id = rating_user_id_entry.get()
        movie_id = rating_movie_id_entry.get()
        rating = rating_value_entry.get()
        if not user_id or not movie_id or not rating:
            raise ValueError("All fields are required.")
        
        payload = [{"userId": int(user_id), "movieId": int(movie_id), "rating": float(rating)}]
        response = requests.post("http://127.0.0.1:8000/add_ratings/", json=payload)
        #add_ratings_result_label.config(text=f"Response: {response.json()}")
        data = response.json()
        
        # Displaying the success message
        if "message" in data:
            add_ratings_result_label.config(text=data["message"])
        else:
            add_ratings_result_label.config(text="Failed to add rating. No message received.")
    except Exception as e:
        add_ratings_result_label.config(text=f"Error: {e}")


"""
# Create main Tkinter window
root = tk.Tk()
root.title("API Testing GUI")

# Frame for Recommend Endpoint
recommend_frame = tk.LabelFrame(root, text="Recommend Endpoint", padx=10, pady=10)
recommend_frame.pack(side="left", padx=10, pady=10, fill="y")

tk.Label(recommend_frame, text="User ID:").pack(anchor="w")
recommend_user_id = tk.Entry(recommend_frame)
recommend_user_id.pack(anchor="w", fill="x")

tk.Label(recommend_frame, text="Model Type:").pack(anchor="w")
model_choice = tk.StringVar(value="knn")
tk.Radiobutton(recommend_frame, text="KNN", variable=model_choice, value="knn").pack(anchor="w")
tk.Radiobutton(recommend_frame, text="SVD", variable=model_choice, value="svd").pack(anchor="w")

tk.Button(recommend_frame, text="Submit", command=recommend).pack(pady=5)
recommend_result_label = tk.Label(recommend_frame, text="")
recommend_result_label.pack(anchor="w")

# Frame for Add Movie Endpoint
add_movie_frame = tk.LabelFrame(root, text="Add Movie Endpoint", padx=10, pady=10)
add_movie_frame.pack(side="left", padx=10, pady=10, fill="y")

tk.Label(add_movie_frame, text="Movie ID:").pack(anchor="w")
movie_id_entry = tk.Entry(add_movie_frame)
movie_id_entry.pack(anchor="w", fill="x")

tk.Label(add_movie_frame, text="Title:").pack(anchor="w")
movie_title_entry = tk.Entry(add_movie_frame)
movie_title_entry.pack(anchor="w", fill="x")

tk.Label(add_movie_frame, text="Genres:").pack(anchor="w")
movie_genres_entry = tk.Entry(add_movie_frame)
movie_genres_entry.pack(anchor="w", fill="x")

tk.Button(add_movie_frame, text="Submit", command=add_movie).pack(pady=5)
add_movie_result_label = tk.Label(add_movie_frame, text="")
add_movie_result_label.pack(anchor="w")

# Frame for Add Ratings Endpoint
add_ratings_frame = tk.LabelFrame(root, text="Add Ratings Endpoint", padx=10, pady=10)
add_ratings_frame.pack(side="left", padx=10, pady=10, fill="y")

tk.Label(add_ratings_frame, text="User ID:").pack(anchor="w")
rating_user_id_entry = tk.Entry(add_ratings_frame)
rating_user_id_entry.pack(anchor="w", fill="x")

tk.Label(add_ratings_frame, text="Movie ID:").pack(anchor="w")
rating_movie_id_entry = tk.Entry(add_ratings_frame)
rating_movie_id_entry.pack(anchor="w", fill="x")

tk.Label(add_ratings_frame, text="Rating:").pack(anchor="w")
rating_value_entry = tk.Entry(add_ratings_frame)
rating_value_entry.pack(anchor="w", fill="x")

tk.Button(add_ratings_frame, text="Submit", command=add_rating).pack(pady=5)
add_ratings_result_label = tk.Label(add_ratings_frame, text="")
add_ratings_result_label.pack(anchor="w")

# Run Tkinter event loop
root.mainloop()
"""
# Create main Tkinter window
root = tk.Tk()
root.title("API Testing GUI")
root.configure(bg="white")

# Custom Font
title_font = font.Font(family="Helvetica", size=12, weight="bold")
label_font = font.Font(family="Arial", size=10)
button_font = font.Font(family="Arial", size=10, weight="bold")

# Frame for Recommend Endpoint
recommend_frame = tk.LabelFrame(
    root, text="Recommend Endpoint", padx=10, pady=10,
    bg="white", fg="#1f77b4", font=title_font
)
recommend_frame.pack(side="left", padx=10, pady=10, fill="y")

tk.Label(recommend_frame, text="User ID:", font=label_font, bg="white").pack(anchor="w")
recommend_user_id = tk.Entry(recommend_frame)
recommend_user_id.pack(anchor="w", fill="x")

tk.Label(recommend_frame, text="Model Type:", font=label_font, bg="white").pack(anchor="w")
model_choice = tk.StringVar(value="knn")
tk.Radiobutton(recommend_frame, text="KNN", variable=model_choice, value="knn", bg="white", font=label_font).pack(anchor="w")
tk.Radiobutton(recommend_frame, text="SVD", variable=model_choice, value="svd", bg="white", font=label_font).pack(anchor="w")

tk.Button(recommend_frame, text="Submit", command=recommend, font=button_font, bg="#1f77b4", fg="white").pack(pady=5)
recommend_result_label = tk.Label(recommend_frame, text="", font=label_font, bg="white")
recommend_result_label.pack(anchor="w")

# Frame for Add Movie Endpoint
add_movie_frame = tk.LabelFrame(
    root, text="Add Movie Endpoint", padx=10, pady=10,
    bg="white", fg="#ff7f0e", font=title_font
)
add_movie_frame.pack(side="left", padx=10, pady=10, fill="y")

tk.Label(add_movie_frame, text="Movie ID:", font=label_font, bg="white").pack(anchor="w")
movie_id_entry = tk.Entry(add_movie_frame)
movie_id_entry.pack(anchor="w", fill="x")

tk.Label(add_movie_frame, text="Title:", font=label_font, bg="white").pack(anchor="w")
movie_title_entry = tk.Entry(add_movie_frame)
movie_title_entry.pack(anchor="w", fill="x")

tk.Label(add_movie_frame, text="Genres:", font=label_font, bg="white").pack(anchor="w")
movie_genres_entry = tk.Entry(add_movie_frame)
movie_genres_entry.pack(anchor="w", fill="x")

tk.Button(add_movie_frame, text="Submit", command=add_movie, font=button_font, bg="#ff7f0e", fg="white").pack(pady=5)
add_movie_result_label = tk.Label(add_movie_frame, text="", font=label_font, bg="white")
add_movie_result_label.pack(anchor="w")

# Frame for Add Ratings Endpoint
add_ratings_frame = tk.LabelFrame(
    root, text="Add Ratings Endpoint", padx=10, pady=10,
    bg="white", fg="#2ca02c", font=title_font
)
add_ratings_frame.pack(side="left", padx=10, pady=10, fill="y")

tk.Label(add_ratings_frame, text="User ID:", font=label_font, bg="white").pack(anchor="w")
rating_user_id_entry = tk.Entry(add_ratings_frame)
rating_user_id_entry.pack(anchor="w", fill="x")

tk.Label(add_ratings_frame, text="Movie ID:", font=label_font, bg="white").pack(anchor="w")
rating_movie_id_entry = tk.Entry(add_ratings_frame)
rating_movie_id_entry.pack(anchor="w", fill="x")

tk.Label(add_ratings_frame, text="Rating:", font=label_font, bg="white").pack(anchor="w")
rating_value_entry = tk.Entry(add_ratings_frame)
rating_value_entry.pack(anchor="w", fill="x")

tk.Button(add_ratings_frame, text="Submit", command=add_rating, font=button_font, bg="#2ca02c", fg="white").pack(pady=5)
add_ratings_result_label = tk.Label(add_ratings_frame, text="", font=label_font, bg="white")
add_ratings_result_label.pack(anchor="w")

# Run Tkinter event loop
root.mainloop()
