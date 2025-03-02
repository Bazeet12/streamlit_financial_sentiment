import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        # Force download regardless of whether the resource exists locally
        nltk.download('punkt')
        nltk.download('stopwords')
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {str(e)}")
        return False

# Ensure resources are downloaded before proceeding
resources_available = download_nltk_resources()

# Preprocessing function
@st.cache_data
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    # Basic text cleaning
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Safe tokenization with error handling
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    except LookupError:
        # Fallback if tokenization fails
        st.warning("NLTK tokenization failed. Using basic space-based tokenization instead.")
        tokens = text.split()
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            # If stopwords also fail
            pass
        return ' '.join(tokens)

# Load the DistilBERT model and tokenizer
@st.cache_resource
def load_distilbert_model():
    try:
        model_path = 'best_model/transformer_model'
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Predict sentiment
def predict_sentiment(text, model, tokenizer):
    processed_text = preprocess_text(text)
    if not processed_text:
        return "Unknown", 0.0
        
    inputs = tokenizer(
        processed_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='tf'
    )
    outputs = model(inputs)
    logits = outputs.logits.numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    prediction = np.argmax(probs)
    sentiment = "Positive" if prediction == 1 else "Negative/Neutral"
    confidence = float(probs[prediction])
    return sentiment, confidence

# Main function
def main():
    st.title("📊 Financial Sentiment Analysis")
    st.markdown("### Powered by DistilBERT")
    
    if not resources_available:
        st.error("Failed to download required NLTK resources. Please check your internet connection.")
        st.info("Alternative solution: Run the following in your terminal before starting the app:\n```\npython -m nltk.downloader punkt stopwords\n```")
        st.stop()
    
    model, tokenizer = load_distilbert_model()
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check model files in 'best_model/transformer_model'.")
        st.stop()
    
    # User input
    st.subheader("Enter text for sentiment analysis")
    user_input = st.text_area("Type or paste financial text:", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
                
                # Display results
                if sentiment == "Positive":
                    st.success(f"Sentiment: {sentiment}")
                else:
                    st.error(f"Sentiment: {sentiment}")
                st.info(f"Confidence: {confidence:.2%}")
                
                with st.expander("View Preprocessed Text"):
                    st.write(preprocess_text(user_input))
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
