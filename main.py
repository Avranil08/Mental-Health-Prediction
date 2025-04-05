import streamlit as st
import torch
import torch.nn as nn
import re
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
import torch.optim as optim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# # Loading the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model architecture and weights
model = BertForSequenceClassification.from_pretrained(
    "Avranil/Mental_Health_Bert",
    torch_dtype="auto",
)
# model.load_state_dict(torch.load("my_model.pth", map_location=device))
model.eval()

# Loading the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Preprocessing function for tweets
def preprocess_tweet(tweet):

    tweet = re.sub(r"@\w+", "", tweet)  # Remove @mentions
    tweet = re.sub(r"#\w+", "", tweet)  # Remove hashtags
    tweet = re.sub(r"http\S+", "", tweet)  # Remove URLs

    # Remove non-alphabetic characters (to keep only words)
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Remove extra spaces
    tweet = " ".join(tweet.split())

    tokens = word_tokenize(tweet)

    stop_words = set(stopwords.words("english"))

    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()

    tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)


# Function to predict mental health status based on tweet input
def predict_mental_health(tweet):
    # Preprocess the tweet before feeding it to the model
    processed_tweet = preprocess_tweet(tweet)

    # Tokenize the input text
    inputs = tokenizer(
        processed_tweet,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Making the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()

    confidence = probs[0][predicted_class].item() * 100

    pred = [
        "Anxiety",
        "Bipolar",
        "Depression",
        "Normal",
        "Personality Disorder",
        "Stress",
        "Suicidal",
    ]

    return (
        f"Mental Health status: {pred[predicted_class]}",
        f"Confidence: {confidence:.2f}%",
    )


# Streamlit App Layout
st.title("Mental Health Prediction from Tweets")


tweet_input = st.text_area("Enter a tweet to predict mental health:", height=150)


if st.button("Predict"):
    if tweet_input:
        result, confidence = predict_mental_health(tweet_input)
        st.write(result)
        st.write(confidence)
    else:
        st.warning("Please enter a tweet to get a prediction.")
