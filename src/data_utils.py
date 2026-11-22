"""Shared data helpers. Owner: Mouaz."""
import pandas as pd


def liked_ratings_with_genres(ratings_df, movies_df):
    """Return only the liked ratings (rating >= 4.0) with a 'genres' text column,
    e.g. 'Action|Sci-Fi', which is the format mine_rules() expects."""
    genre_cols = movies_df.columns.drop('movieId')
    genre_text = (movies_df.set_index('movieId')[genre_cols].apply(lambda row: '|'.join(genre_cols[row.values]), axis=1))
    liked = ratings_df[ratings_df['rating'] >= 4.0]
    return liked.assign(genres=liked['movieId'].map(genre_text))