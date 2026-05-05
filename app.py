import pickle
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.header('Book Recommender System Using Machine Learning')
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
content_matrix = pickle.load(open('artifacts/content_matrix.pkl', 'rb'))
content_titles = pickle.load(open('artifacts/content_titles.pkl', 'rb'))
books_meta = pickle.load(open('artifacts/books_meta.pkl', 'rb'))

books_meta = books_meta.drop_duplicates('title')
image_url_by_title = dict(zip(books_meta['title'], books_meta['image_url']))
content_title_to_idx = {title: idx for idx, title in enumerate(content_titles)}


def get_poster_urls(titles):
    poster_urls = []
    for title in titles:
        poster_urls.append(image_url_by_title.get(title, ''))
    return poster_urls


def recommend_book(book_name, top_k=5, alpha=0.7, collab_k=50, content_k=50):
    scores = {}

    if book_name in book_pivot.index:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        n_neighbors = min(collab_k + 1, len(book_pivot))
        distances, suggestion = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=n_neighbors
        )

        for idx, dist in zip(suggestion[0], distances[0]):
            title = book_pivot.index[idx]
            if title == book_name:
                continue
            scores[title] = scores.get(title, 0.0) + alpha * (1.0 - dist)

    if book_name in content_title_to_idx:
        cidx = content_title_to_idx[book_name]
        sims = cosine_similarity(content_matrix[cidx], content_matrix).flatten()
        top_idx = np.argsort(-sims)[1:content_k + 1]
        for idx in top_idx:
            title = content_titles[idx]
            if title == book_name:
                continue
            scores[title] = scores.get(title, 0.0) + (1.0 - alpha) * sims[idx]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended = [title for title, _ in ranked[:top_k]]
    poster_url = get_poster_urls(recommended)
    return recommended, poster_url



selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)
    if not recommended_books:
        st.warning('No recommendations available for this book yet.')
    else:
        cols = st.columns(min(5, len(recommended_books)))
        for i, col in enumerate(cols):
            col.text(recommended_books[i])
            if poster_url[i]:
                col.image(poster_url[i])