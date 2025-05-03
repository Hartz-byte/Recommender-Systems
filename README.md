![Machine Learning](https://img.shields.io/badge/Machine_Learning-ML-blue?logo=python)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)
![Content-Based](https://img.shields.io/badge/Recommendation-Content--Based-informational)
![TF-IDF](https://img.shields.io/badge/Vectorizer-TF--IDF-critical)
![Cosine Similarity](https://img.shields.io/badge/Similarity-Cosine-blueviolet)
![PCA](https://img.shields.io/badge/Dimensionality_Reduction-PCA-important)
![WordCloud](https://img.shields.io/badge/WordCloud-Visualization-lightgrey)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Movie Recommendation System with NLP, TF-IDF & PCA

A content-based movie recommendation system that suggests similar movies based on genres, keywords, cast, and director using **Natural Language Processing (NLP)** techniques. The project includes interactive visualizations like **PCA projections** and **word clouds** to better understand relationships between movies.

---

## Features

- **Content-Based Recommendations** (no user ratings needed)
- **TF-IDF Vectorization** of movie metadata
- **Dimensionality Reduction** using PCA for 2D visualization
- **Scatter Plot** of movie clusters based on content
- **Word Cloud** representing most frequent keywords across movies
- Clean and interpretable code with helper functions and modular structure

---

## Tech Stack

| Category              | Technology                |
|-----------------------|---------------------------|
| Programming Language  | Python 3                  |
| Libraries & Tools     | Pandas, NumPy, Scikit-learn |
| Visualization         | Matplotlib, Seaborn, WordCloud |
| Dataset               | TMDb 5000 Movie Dataset from Kaggle |

---

## Dataset

Dataset used: [TMDb 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

- `tmdb_5000_movies.csv` — Movie metadata including genres, keywords, overview
- `tmdb_5000_credits.csv` — Cast and crew details for each movie

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install Dependencies
You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
```

### 3. Download Dataset
Place the following CSV files inside the project directory:

tmdb_5000_movies.csv

tmdb_5000_credits.csv

You can download these from Kaggle.

---

## Visualizations
- PCA Plot of Movies
  Displays a 2D scatter plot using PCA to reduce the high-dimensional TF-IDF matrix. Each point represents a movie, colored by its genre. Similar movies are closer together.


- Word Cloud
  The word cloud visualizes the most frequent terms from all movie metadata (overview, genres, keywords, cast, and director), helping to identify trends in the dataset.

---

## How It Works
1. Data Cleaning & Preprocessing
  - Parse JSON-like strings for genres, cast, crew, and keywords.
  - Extract key features: top 3 cast members and director.

2. Feature Engineering
  - Merge overview, genres, keywords, cast, and crew into a single tags column.
  - Normalize the text: lowercase, no spaces, and stopword removal.

3. Vectorization
  - Use TfidfVectorizer to convert text into feature vectors.
  - Each movie is represented by a 5000-dimensional sparse vector.

4. Similarity Calculation
  - Compute pairwise cosine similarity between movies.

5. PCA for Visualization
  - Reduce 5000D TF-IDF vectors to 2D using Principal Component Analysis (PCA).
  - Visualize the content-based proximity between movies.

6. Recommendation Function
  - Takes a movie title as input.
  - Returns the top 5 most similar movies based on cosine similarity.

---

## Example Usage
```bash
recommend('The Dark Knight')
```

```bash
Top 5 Recommendations for 'The Dark Knight':
 - The Dark Knight Rises
 - Batman Returns
 - Batman Begins
 - Batman Forever
 - Batman: The Dark Knight Returns, Part 2
```

## Learnings
- How to engineer text-based features from structured and semi-structured data.
- Implemented TF-IDF from scratch for a recommender system.
- Learned to apply PCA for visualizing high-dimensional text vectors.
- Hands-on with cosine similarity, vector math, and NLP in Python.

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---
