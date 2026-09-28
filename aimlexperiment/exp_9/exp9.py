import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data and combine genre and description for comparison.
movies = pd.read_csv("database.csv")
movies["Features"] = movies["Genre"] + " " + movies["Description"]

# Convert each movie's features into TF-IDF values.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies["Features"])

# Ask for a genre and find matching movies in the database.

genre = input("Enter a genre: ").strip()
genres = movies["Genre"].drop_duplicates()
genre_match = genres[genres.str.lower() == genre.lower()]

if len(genre_match) == 0:
    print("Genre not found. Available genres:")
    print(", ".join(genres))
else:
    selected_genre = genre_match.iloc[0]
    matching_movies = movies.index[movies["Genre"] == selected_genre]

    # Compare matching movies and show up to five from the selected genre.
    genre_similarity = cosine_similarity(
        tfidf_matrix[matching_movies], tfidf_matrix[matching_movies]
    )
    average_similarity = genre_similarity.mean(axis=1)
    ranked_movies = sorted(
        zip(matching_movies, average_similarity),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    print(f"\nMovies in the {selected_genre} genre:")
    for index, score in ranked_movies:
        print(movies.loc[index, "Movie"])

    if len(ranked_movies) < 5:
        print(f"Only {len(ranked_movies)} movies in this genre are available.")