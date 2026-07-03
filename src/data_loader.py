"""
Loads the dair-ai/emotion dataset.
Reused later by train.py (Phase 5/6) so training and any future
re-loading always pull data the exact same way.
"""
from datasets import load_dataset


def load_data():
    """Downloads (or loads from local cache) the dataset."""
    dataset = load_dataset("dair-ai/emotion")                            
    return dataset


if __name__ == "__main__":
    dataset = load_data()
    print(dataset)

    # Pull label names from the dataset's own metadata rather than
    # hardcoding them — hardcoding is a common beginner mistake that
    # breaks silently if the dataset version ever changes label order.
    label_names = dataset["train"].features["label"].names
    print("Label mapping:", {i: name for i, name in enumerate(label_names)})
    # --- Exploration ---
       
    import pandas as pd
    train_df = dataset["train"].to_pandas()

    print("\nShape:", train_df.shape)
    print("\nMissing values:\n", train_df.isnull().sum())
    print("\nClass distribution:\n", train_df["label"].value_counts())
    print("\nSample records:\n", train_df.sample(5, random_state=42))