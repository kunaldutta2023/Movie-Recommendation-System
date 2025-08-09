🎬 Movie Recommendation System

A Machine Learning-based recommendation engine that suggests movies to users based on their preferences. Built with Python, Pandas, Scikit-learn, and deployed on AWS EC2 with dataset stored in AWS S3.

📌 Features

Personalized Recommendations – Suggests movies similar to a given movie.

Content-Based Filtering – Uses movie metadata for recommendations.

AWS Deployment – Publicly hosted on AWS EC2 instance.

Public Dataset Access – Movies & ratings data stored in AWS S3 for easy access.


🛠️ Tech Stack

Languages: Python, HTML, CSS

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib

Cloud: AWS EC2, AWS S3

Version Control: Git, GitHub


🚀 Installation & Usage

1. Clone the Repository

git clone https://github.com/kunaldutta2023/Movie-Recommendation-System.git
cd Movie-Recommendation-System


2. Install Dependencies

pip install -r requirements.txt


3. Run the Application

python app.py


4. Access the App
Open http://localhost:5000 in your browser.



📊 How It Works

1. Data Preprocessing – Loads datasets from AWS S3.


2. Model Training – Uses NearestNeighbors to find similar movies.


3. Recommendation Engine – Returns top N similar movies based on a given title.


👨‍💻 Author

Kunal Dutta

GitHub: @kunaldutta2023

LinkedIn: @kunal-dutta-it2023
