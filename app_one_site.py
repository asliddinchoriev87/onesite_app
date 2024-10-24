import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained NMF model, vectorizer, and summarization model
with open('nmf_model.pkl', 'rb') as model_file:
    nmf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('summarization_model.pkl', 'rb') as sum_file:
    summarization_model = pickle.load(sum_file)

# Load the news data from the .pkl file
st.title("News Article Categorization and Summarization")

if os.path.exists('news_data.pkl'):
    with open('news_data.pkl', 'rb') as f:
        articles = pickle.load(f)

    st.write(f"Number of articles found: {len(articles)}")

    topics_to_categories = {0: "Sports", 1: "Politics", 2: "Business", 3: "Entertainment", 4: "Technology"}

    # Iterate through each article and display results
    for index, article in enumerate(articles):
        title = article['Title']
        text = article['Text Content']

        st.subheader(f"Article {index + 1}: {title}")
        
        # Combine title and content text
        full_text = title + " " + text

        # Step 1: Transform the text using the TF-IDF vectorizer
        text_tfidf = vectorizer.transform([full_text])

        # Step 2: Get topic distribution from the NMF model
        topic_distribution = nmf_model.transform(text_tfidf)

        # Step 3: Get the most relevant topic
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
    st.error("No news_data.pkl file found. Please upload it.")
