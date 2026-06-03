import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = pd.DataFrame({
    'title': [
        'Inception',
        'Interstellar',
        'The Dark Knight',
        'Titanic',
        'Avatar',
        'The Avengers',
        'Iron Man',
        'Doctor Strange'
    ],
    'genre': [
        'Sci-Fi Thriller',
        'Sci-Fi Adventure',
        'Action Crime',
        'Romance Drama',
        'Sci-Fi Adventure',
        'Action Sci-Fi',
        'Action Sci-Fi',
        'Fantasy Sci-Fi'
    ]
})

# Convert genres into vectors
cv = CountVectorizer()
vectors = cv.fit_transform(movies['genre'])

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):
    if movie_name not in movies['title'].values:
        print("Movie not found!")
        return

    movie_index = movies[movies['title'] == movie_name].index[0]

    distances = list(enumerate(similarity[movie_index]))

    recommended_movies = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )

    print(f"\nMovies similar to {movie_name}:\n")

    for movie in recommended_movies[1:6]:
        print(movies.iloc[movie[0]].title)

# User input
movie = input("Enter a movie name: ")
recommend(movie)
