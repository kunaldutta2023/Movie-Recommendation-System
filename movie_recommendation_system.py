import pandas as pd
import numpy as np
import boto3
from io import StringIO
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 🚀 AWS S3 CONFIGURATION
s3_bucket = "my-movie-recommendation-dataset"  # Replace with your bucket name
movies_file = "movies.csv"  # Change if filename differs
ratings_file = "ratings.csv"

# 🔹 Initialize S3 Client
s3 = boto3.client('s3')

# 🔹 Function to Read CSV from S3
def read_csv_from_s3(bucket, file_key):
    try:
        response = s3.get_object(Bucket=bucket, Key=file_key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        print(f"✅ Successfully loaded {file_key} from S3. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_key}: {e}")
        return None

# 🔹 Load Datasets
movies = read_csv_from_s3(s3_bucket, movies_file)
ratings = read_csv_from_s3(s3_bucket, ratings_file)

# 🛠️ CHECK DATA INTEGRITY
if movies is None or ratings is None:
    print("❌ Dataset loading failed. Check S3 permissions or file paths.")
    exit()

print("📊 Movies Data Preview:\n", movies.head())
print("📊 Ratings Data Preview:\n", ratings.head())

# 🏗️ Create User-Movie Ratings Matrix
final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)
print("✅ Created Pivot Table: ", final_dataset.shape)

# 🗑️ Remove Movies with <10 Votes
no_user_voted = ratings.groupby("movieId")['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]

# 🗑️ Remove Users Who Rated <50 Movies
no_movies_voted = ratings.groupby("userId")['rating'].agg('count')
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# 🔹 Check Processed Data
print("✅ Processed Dataset Shape:", final_dataset.shape)

# 🔹 Convert to Sparse Matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# 🔍 Train KNN Model for Movie Recommendations
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
print("✅ KNN Model Trained!")

# 🔥 RECOMMENDATION FUNCTION
def get_recommendation(movie_name):
    print(f"🔍 Searching for movie: {movie_name}")
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]
    
    if len(movie_list) == 0:
        print("❌ Movie not found in dataset")
        return ["Movie not found..."]

    movie_idx = movie_list.iloc[0]['movieId']
    print(f"✅ Found Movie ID: {movie_idx}")

    try:
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
    except IndexError:
        print("❌ Movie ID not found in final dataset")
        return ["Movie not found..."]

    print(f"🔍 Finding similar movies for ID {movie_idx}...")
    distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=6)

    recommendations = []
    for idx in indices.flatten()[1:]:
        rec_movie_id = final_dataset.iloc[idx]['movieId']
        rec_movie = movies[movies['movieId'] == rec_movie_id]["title"].values[0]
        recommendations.append(rec_movie)

    print(f"🎬 Recommended Movies: {recommendations}")
    return recommendations

# 🎬 Test Recommendation Function
print(get_recommendation("Avatar"))