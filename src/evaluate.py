"""
Full evaluation of the selected model (Linear SVM) on the untouched test set.
This is the ONE time test data gets used — confirms real generalization
without any risk of decisions being indirectly tuned to it.
"""
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from features import load_and_clean

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def evaluate():
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_and_clean()

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Wrap LinearSVC so it outputs probabilities (needed for Phase 8 confidence scores)
    base_svm = LinearSVC()
    model = CalibratedClassifierCV(base_svm, cv=5)
    model.fit(X_train, train_labels)

    preds = model.predict(X_test)

    print("Test Accuracy:", accuracy_score(test_labels, preds))
    print("Test Macro F1:", f1_score(test_labels, preds, average="macro"))
    print("\nClassification Report:\n")
    print(classification_report(test_labels, preds, target_names=LABEL_NAMES))

    cm = confusion_matrix(test_labels, preds)
    plot_confusion_matrix(cm)

    return model, vectorizer


def plot_confusion_matrix(cm):
    """Plain matplotlib heatmap — avoids the seaborn/scipy import issue from Phase 3."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(6), LABEL_NAMES, rotation=45)
    plt.yticks(range(6), LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Linear SVM (Test Set)")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.close()
    print("\nSaved: reports/figures/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()