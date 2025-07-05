import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load Dataset
df = pd.read_csv(r"netflix_titles.csv")

# Preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words("english"))
    return ' '.join([word for word in text.split() if word not in stop_words])

df['description'] = df['description'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)
df['combined'] = df['title'] + ' ' + df['description'] + ' ' + df['listed_in'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def recommend(title_input, top_n=5):
    title_input = clean_text(title_input)
    input_vec = tfidf.transform([title_input])
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    indices = sim_scores.argsort()[-top_n-1:][::-1]
    recommended = df.iloc[indices][['title', 'description', 'listed_in', 'type', 'release_year']]
    return recommended[recommended['title'] != title_input].head(top_n)

# Streamlit UI
st.set_page_config(page_title="Netflix Recommender", layout="wide")
st.title("🎬 Netflix Movie & TV Show Recommender")
st.write("Enter a movie or show name and get similar recommendations using NLP!")

user_input = st.text_input("Enter a Movie/Show Title:", "Stranger Things")
if user_input:
    results = recommend(user_input)
    if not results.empty:
        st.subheader("Recommended Titles:")
        for idx, row in results.iterrows():
            st.markdown(f"### {row['title'].title()}")
            st.markdown(f"**Type:** {row['type']}")
            st.markdown(f"**Genre:** {row['listed_in']}")
            st.markdown(f"**Release Year:** {row['release_year']}")
            st.markdown(f"**Description:** {row['description'].capitalize()}")
            st.markdown("---")
    else:
        st.warning("No similar titles found. Try another title.")