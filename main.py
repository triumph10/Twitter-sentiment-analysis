import re
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to clean text
def clean_text(text):
    text = re.sub(r'@[\w]*', '', text)
    text = re.sub('[^a-zA-Z#]', ' ', text)
    text = ' '.join([w for w in text.split() if len(w) > 3])
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Load the model
model_rnn = load_model('LSTM_emotion_model.h5')

# Assuming the tokenizer was saved during training, load it
# Example: tokenizer was saved using pickle
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Assuming you know the sequence length used during training
sequence_length = 117  # Replace with your actual sequence length

# Streamlit App
st.title("Track your mOOd")

input_text = st.text_area("Enter a text:")
cleaned_text = clean_text(input_text)

if st.button("Classify"):
    if len(cleaned_text) == 0:
        st.write("Please enter some text.")
    else:
        # Tokenize and pad the cleaned text
        text_sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(text_sequence, maxlen=sequence_length)

        # Make a prediction
        prediction = model_rnn.predict(padded_sequence)
        predicted_label = prediction.argmax(axis=-1)

        labels = {0: "Anger", 1: "Fear", 2: "Joy", 3: "Neutral", 4: "Sadness"}
        st.write(f"Sentiment: {labels[predicted_label[0]]}")
