"""
Feature engineering — TF-IDF vectorization.
Fit ONLY on training data; validation/test are transformed using the
already-fitted vectorizer (prevents data leakage — see Phase 0).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from data_loader import load_data
from preprocessing import clean_text


def load_and_clean():
    """Loads all 3 splits and applies the identical cleaning pipeline to each."""
    dataset = load_data()
    train_texts = [clean_text(t) for t in dataset["train"]["text"]]
    val_texts = [clean_text(t) for t in dataset["validation"]["text"]]
    test_texts = [clean_text(t) for t in dataset["test"]["text"]]

    train_labels = dataset["train"]["label"]
    val_labels = dataset["validation"]["label"]
    test_labels = dataset["test"]["label"]

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


if __name__ == "__main__":
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_and_clean()

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)   # fit + transform, train only
    X_val = vectorizer.transform(val_texts)             # transform only, no fit
    X_test = vectorizer.transform(test_texts)           # transform only, no fit

    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Train matrix shape:", X_train.shape)
    print("Validation matrix shape:", X_val.shape)
    print("Test matrix shape:", X_test.shape)

    example_idx = 0
    row = X_train[example_idx].toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    top_indices = row.argsort()[-5:][::-1]
    print(f"\nExample tweet: {train_texts[example_idx]}")
    print("Top TF-IDF words:", [(feature_names[i], round(row[i], 3)) for i in top_indices if row[i] > 0])