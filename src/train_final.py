"""
Trains the final selected model (Linear SVM, calibrated) ONE TIME and
persists both the model and TF-IDF vectorizer to /models using joblib.
Everything after this loads these saved artifacts instead of retraining.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from features import load_and_clean


def train_and_save():
    (train_texts, train_labels), _, _ = load_and_clean()

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)

    # random_state=42 added here for reproducibility — without it, LinearSVC's
    # internal solver can produce very slightly different results run-to-run.
    base_svm = LinearSVC(random_state=42)
    model = CalibratedClassifierCV(base_svm, cv=5)
    model.fit(X_train, train_labels)

    joblib.dump(model, "models/emotion_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("Saved: models/emotion_model.pkl")
    print("Saved: models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    train_and_save()