"""
Prediction system: predict_emotion(text) -> (label, confidence)
NOTE: retrains the model on each run for now — Phase 9 will persist the
trained model + vectorizer with joblib so this becomes instant.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from features import load_and_clean
from preprocessing import clean_text

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def build_model():
    (train_texts, train_labels), _, _ = load_and_clean()
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)

    base_svm = LinearSVC()
    model = CalibratedClassifierCV(base_svm, cv=5)
    model.fit(X_train, train_labels)

    return model, vectorizer


def predict_emotion(text, model, vectorizer):
    """
    Takes raw text, applies the SAME preprocessing + vectorizer used in
    training (transform only, never fit), returns (label, confidence).
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]

    predicted_idx = probs.argmax()
    label = LABEL_NAMES[predicted_idx]
    confidence = probs[predicted_idx]

    return label, confidence


if __name__ == "__main__":
    model, vectorizer = build_model()

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