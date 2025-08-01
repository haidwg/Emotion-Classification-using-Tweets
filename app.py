# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load fine-tuned model
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_emotion_model")
    tokenizer = AutoTokenizer.from_pretrained("models/distilbert_emotion_model")
    return model.eval(), tokenizer

model, tokenizer = load_model()
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

st.title("Emotion Detection 🧠💬")
st.write("Enter any sentence and get the top 3 emotions predicted by the model.")

custom_text = st.text_area("Input Text", "I'm so excited for the weekend!")

if st.button("Predict"):
    if custom_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(custom_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze().numpy()

        # Get top 3
        top_indices = np.argsort(probs)[-3:][::-1]

        st.subheader("Top 3 Predicted Emotions:")
        for idx in top_indices:
            st.write(f"**{label_names[idx]}**: {probs[idx]:.4f}")
