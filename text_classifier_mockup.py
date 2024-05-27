import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Set page title and description
st.title("Customer Review Sentiment Analysis")
st.write("""
This tool performs sentiment analysis on customer reviews from an uploaded dataset. It uses spaCy for preprocessing and TextBlob for sentiment analysis.
""")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload customer review dataset (CSV file)")

if uploaded_file is not None:
    # Read dataset
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Allow user to select the column containing the reviews
    st.sidebar.header("Select Column")
    text_column = st.sidebar.selectbox("Select the column containing the reviews", df.columns)

    # Perform sentiment analysis and count positive, negative, and neutral reviews
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    for review in df[text_column]:
        # Convert review to string
        review = str(review)

        # Preprocess the text using spaCy
        doc = nlp(review)
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

        # Perform sentiment analysis using TextBlob
        text_blob = TextBlob(" ".join(tokens))
        sentiment_score = text_blob.sentiment.polarity

        # Classify sentiment based on the score
        if sentiment_score > 0.2:
            positive_count += 1
        elif sentiment_score < -0.2:
            negative_count += 1
        else:
            neutral_count += 1

    st.header("Review Counts")
    st.write(f"Positive Reviews: {positive_count}")
    st.write(f"Negative Reviews: {negative_count}")
    st.write(f"Neutral Reviews: {neutral_count}")

    # Option to view comments
    if st.checkbox("View Comments"):
        for review in df[text_column]:
            st.write(f"Comment: {review}")
