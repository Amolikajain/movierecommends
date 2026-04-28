import streamlit as st
import pickle
import pandas as pd
import difflib

# ----------------- LOAD FILES -----------------
movie_df = pickle.load(open('movie_df.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
indices = pickle.load(open('indices.pkl', 'rb'))

# ----------------- RECOMMEND FUNCTION -----------------
from sklearn.metrics.pairwise import cosine_similarity

def recommend(movie_name):
    if movie_name not in indices:
        return []

    idx = indices[movie_name]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]
    return movie_df.iloc[movie_indices]

# ----------------- SPELL CORRECTION -----------------
def correct_spelling(movie_name):
    matches = difflib.get_close_matches(movie_name, movie_df['title'].tolist(), n=1, cutoff=0.5)
    return matches[0] if matches else None

# ----------------- UI -----------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .movie-card {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)

# ----------------- INPUT -----------------
col1, col2 = st.columns([2,1])

with col1:
    movie_input = st.text_input("Type a movie name:")

with col2:
    selected_movie = st.selectbox("Or select a movie:", movie_df['title'].values)

# ----------------- BUTTON -----------------
if st.button("Recommend 🎥"):
    
    movie_name = movie_input if movie_input else selected_movie

    # Handle spelling mistakes
    if movie_name not in indices:
        corrected = correct_spelling(movie_name)
        if corrected:
            st.warning(f"Did you mean: {corrected}? Showing results for it.")
            movie_name = corrected
        else:
            st.error("Movie not found!")
            st.stop()

    recommendations = recommend(movie_name)

    st.subheader("Top Recommendations 🔥")

    cols = st.columns(5)

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
                <div class="movie-card">
                    <h4>{recommendations.iloc[i]['title']}</h4>
                </div>
            """, unsafe_allow_html=True)