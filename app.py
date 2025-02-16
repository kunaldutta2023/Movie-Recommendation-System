from flask import Flask, render_template, request
import pandas as pd
from movie_recommendation_system import get_recommendation  # Import your ML function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None

    if request.method == "POST":
        movie_name = request.form.get("movie")
        recommendations = get_recommendation(movie_name)

        if isinstance(recommendations, str):
            recommendations = [recommendations]
        else:
            recommendations = ["Movie not found..."]


    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
