"""
Trains and compares three baseline models on TF-IDF features from Phase 5.
This phase gets a first-pass comparison; full evaluation (confusion matrix,
classification report) happens in Phase 7.
"""
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from features import load_and_clean


def train_and_compare():
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_and_clean()

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }

    print(f"{'Model':<25}{'Accuracy':<12}{'Macro F1':<12}{'Train Time (s)'}")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, train_labels)
        train_time = time.time() - start

        preds = model.predict(X_val)
        acc = accuracy_score(val_labels, preds)
        macro_f1 = f1_score(val_labels, preds, average="macro")

        print(f"{name:<25}{acc:<12.4f}{macro_f1:<12.4f}{train_time:.2f}")

    return vectorizer, models


if __name__ == "__main__":
    train_and_compare()