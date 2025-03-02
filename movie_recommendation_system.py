import pandas as pd
import numpy as np
import boto3
from io import StringIO
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Set up S3 bucket details
s3_bucket = "my-movie-recommendation-dataset"
movies_file = "movies.csv"
ratings_file = "ratings.csv"

# Initialize boto3 client (Ensure EC2 IAM role has S3 access)
s3 = boto3.client('s3')

# Function to read CSV from S3
def read_csv_from_s3(bucket, file_key):
    response = s3.get_object(Bucket=bucket, Key=file_key)
    return pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

# Load datasets from S3
movies = read_csv_from_s3(s3_bucket, movies_file)
ratings = read_csv_from_s3(s3_bucket, ratings_file)

# Display first few rows
print(movies.head())
print(ratings.head())

# Pivot ratings into a matrix
final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)

# Remove movies with less than 10 votes
no_user_voted = ratings.groupby("movieId")['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]

# Remove users who rated fewer than 50 movies
no_movies_voted = ratings.groupby("userId")['rating'].agg('count')
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Convert to sparse matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Train KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Recommendation function
def get_recommendation(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]
    
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        try:
            movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        except IndexError:
            return ["Movie not found..."]

        distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=6)
        recommendations = []

        for idx in indices.flatten()[1:]:
            rec_movie_id = final_dataset.iloc[idx]['movieId']
            rec_movie = movies[movies['movieId'] == rec_movie_id]["title"].values[0]
            recommendations.append(rec_movie)

        return recommendations
    else:
        return ["Movie not found..."]

# Test recommendation function
print(get_recommendation("Avatar"))