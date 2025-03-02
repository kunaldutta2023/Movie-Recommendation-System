from flask import Flask, render_template, request
import pandas as pd
import boto3
from movie_recommendation_system import get_recommendation  # Import ML function

# S3 Configuration
S3_BUCKET = "my-movie-recommendation-dataset"
s3 = boto3.client("s3")

def load_dataset():
    """Download the latest dataset from S3 and load it into Pandas"""
    s3.download_file(S3_BUCKET, "movies.csv", "movies.csv")
    s3.download_file(S3_BUCKET, "ratings.csv", "ratings.csv")

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_dataset()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        movie_name = request.form.get("movie")
        recommendations = get_recommendation(movie_name)

        if not recommendations:
            recommendations = ["Movie not found..."]

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # Disable debug for production
