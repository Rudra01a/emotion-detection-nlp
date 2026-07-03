"""
Exploratory Data Analysis — class distribution and word frequency.
Saves plots to reports/figures/ (no inline display, since we're script-based).
"""
import matplotlib.pyplot as plt
from collections import Counter
from data_loader import load_data

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def plot_class_distribution(train_df):
       """Bar chart of how many tweets belong to each emotion."""
       counts = train_df["label"].value_counts().sort_index()
       plt.figure(figsize=(8, 5))
       plt.bar(LABEL_NAMES, counts.values, color="#4C72B0")
       plt.title("Emotion Class Distribution (Train Set)")
       plt.xlabel("Emotion")
       plt.ylabel("Count")
       plt.xticks(rotation=30)
       plt.tight_layout()
       plt.savefig("reports/figures/class_distribution.png")
       plt.close()
       print("Saved: reports/figures/class_distribution.png")

def plot_top_words_per_emotion(train_df, top_n=10):
    """For each emotion, find its most frequent words (rough, pre-cleaning)."""
    for label_id, label_name in enumerate(LABEL_NAMES):
        texts = train_df[train_df["label"] == label_id]["text"]
        words = " ".join(texts).split()
        common = Counter(words).most_common(top_n)
        print(f"\nTop words for '{label_name}':", common)


if __name__ == "__main__":
    dataset = load_data()
    train_df = dataset["train"].to_pandas()

    plot_class_distribution(train_df)
    plot_top_words_per_emotion(train_df)