"""
Streamlit app — live demo of the emotion detection model.
Reuses the exact predict_emotion() function from Phase 8/9 — no new
ML logic here, just a UI wrapper around the already-verified pipeline.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from predict import load_artifacts, predict_emotion
from preprocessing import clean_text

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJIS = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}

st.set_page_config(page_title="Emotion Detector", page_icon="🎭")


@st.cache_resource
def get_model():
    """Loads the model+vectorizer once and caches across the session."""
    return load_artifacts()


model, vectorizer = get_model()

st.title("🎭 Emotion Detection from Text")
st.write("Enter a sentence and the model will predict its dominant emotion.")

text = st.text_input("Your sentence:", placeholder="e.g. I can't believe how happy I am today!")

if text:
    label, confidence = predict_emotion(text, model, vectorizer)
    st.markdown(f"### {EMOJIS.get(label, '')} Predicted emotion: **{label}**")
    st.write(f"Confidence: {confidence:.1%}")

    # Show probability across all 6 classes, not just the winner
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]
    prob_df = pd.DataFrame({"Emotion": LABEL_NAMES, "Probability": probs}).set_index("Emotion")
    st.bar_chart(prob_df)

st.markdown("---")
st.caption("Model: TF-IDF + Linear SVM (calibrated) · Trained on dair-ai/emotion · 89.6% test accuracy")