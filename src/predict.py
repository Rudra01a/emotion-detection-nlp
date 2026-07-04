"""
Prediction system: predict_emotion(text) -> (label, confidence)
Loads the saved model + vectorizer from Phase 9 — no retraining.
This is the Phase 0 inference pipeline, now fully realized.
"""
import joblib
from preprocessing import clean_text

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def load_artifacts():
    model = joblib.load("models/emotion_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


def predict_emotion(text, model, vectorizer):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]

    predicted_idx = probs.argmax()
    label = LABEL_NAMES[predicted_idx]
    confidence = probs[predicted_idx]

    return label, confidence


if __name__ == "__main__":
    model, vectorizer = load_artifacts()

    test_sentences = [
        "I just got promoted at work, I can't stop smiling!",
        "I miss my family so much, it's been a hard week.",
        "How dare you talk to me like that, I'm furious.",
        "I'm scared of what might happen tomorrow.",
        "I love spending time with you every single day.",
        "Wow, I did not see that coming at all!",
        "Nothing seems to be going right for me lately.",
        "This is the best day of my life so far.",
        "I can't believe he lied to me again.",
        "The sudden noise made my heart skip a beat.",
    ]

    for sentence in test_sentences:
        label, confidence = predict_emotion(sentence, model, vectorizer)
        print(f"Text: {sentence}")
        print(f"Predicted: {label}  (confidence: {confidence:.2%})\n")