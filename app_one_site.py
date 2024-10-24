import streamlit as st
import pickle
import pandas as pd
from newspaper import Article  # For web scraping
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained NMF model, vectorizer, and summarization model
with open('nmf_model.pkl', 'rb') as model_file:
    nmf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('summarization_model.pkl', 'rb') as sum_file:
    summarization_model = pickle.load(sum_file)

# Function to scrape news data from a given URL
def fetch_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
        text = article.text
        return title, text
    except Exception as e:
        st.error(f"Error fetching the article: {e}")
        return None, None

# Streamlit app layout
st.title("News Article Categorization and Summarization")

# User input: URL for a news article
url = st.text_input("Enter the URL of a news article:")

if url:
    title, text = fetch_article(url)

    if title and text:
        st.write(f"**Title:** {title}")

        # Combine title and content text
        full_text = title + " " + text

        # Step 1: Transform the text using the TF-IDF vectorizer
        text_tfidf = vectorizer.transform([full_text])

        # Step 2: Get topic distribution from the NMF model
        topic_distribution = nmf_model.transform(text_tfidf)

        # Step 3: Get the most relevant topic (category)
        topics_to_categories = {0: "Sports", 1: "Politics", 2: "Business", 3: "Entertainment", 4: "Technology"}
        predicted_topic = topic_distribution.argmax()
        predicted_category = topics_to_categories.get(predicted_topic, "Unknown")

        # Step 4: Summarize the article
        summary = summarization_model.summarizer(text, max_length=40, min_length=20, do_sample=False)

        # Display results
        st.write(f"**Predicted Category:** {predicted_category}")
        st.write(f"**Summary:** {summary[0]['summary_text']}")

        st.write("**Top 5 Keywords:**")
        for keyword, score in summarization_model.extract_top_keywords(text):
            st.write(f"{keyword} : {score:.2f}")

        st.write("-" * 40)
    else:
        st.error("Failed to retrieve the article. Please check the URL.")
else:
    st.write("Please enter a URL to analyze a news article.")
