import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load model and tokenizer
model = load_model("headline_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_length = 20  # Must match your training config

# --- Helper Functions ---

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_headline(seed_text, next_words=10, temperature=1.0):
    last_word = None
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        predicted_index = sample_with_temperature(preds, temperature)
        output_word = tokenizer.index_word.get(predicted_index, "")
        if output_word and output_word != "<OOV>" and output_word != last_word:
            seed_text += " " + output_word
            last_word = output_word
    return seed_text

# Optional: semantic filter using simple keyword checks
def is_semantically_safe(text):
    unsafe_keywords = ["death", "violence", "attack", "war", "tragedy", "politics"]
    return not any(keyword in text.lower() for keyword in unsafe_keywords)

def generate_safe_headline(seed_text, next_words=10, temperature=1.0):
    for _ in range(10):
        headline = generate_headline(seed_text, next_words, temperature)
        if is_semantically_safe(headline):
            return headline
    return "Could not generate a safe headline."

# --- Streamlit App ---

st.title("📰 NYT Headline Generator")
st.write("Generate news headlines using your custom-trained LSTM model. You can optionally filter out unsafe topics.")

seed_text = st.text_input("Enter a seed phrase:", "tech stocks")
next_words = st.slider("Number of words to generate", 5, 20, 10)
temperature = st.slider("Temperature", 0.5, 1.5, 1.0)
safe_mode = st.checkbox("Generate safe headline only")

if st.button("Generate Headline"):
    with st.spinner("Generating..."):
        if safe_mode:
            headline = generate_safe_headline(seed_text, next_words, temperature)
        else:
            headline = generate_headline(seed_text, next_words, temperature)
        st.success("Generated Headline:")
        st.write(headline)
