"""
Text preprocessing functions.
Filled in during Phase 4 — must produce identical output whether
called during training or during inference (see Phase 0 architecture).
"""
"""
Text preprocessing pipeline.
This EXACT function is reused during both training (fit) and inference
(predict) — see Phase 0 architecture. Consistency here is what prevents
train/inference mismatch.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# One-time downloads — safe to call every run, skips automatically if already present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# Standard English stopwords MINUS negation words.
# Removing "not"/"no"/"never" would flip meaning (e.g. "not happy" -> "happy") —
# a common beginner mistake, avoided here deliberately.
NEGATIONS = {"not", "no", "nor", "never", "none", "n't", "cannot", "couldn't",
             "shouldn't", "wouldn't", "won't", "didn't", "doesn't", "isn't",
             "wasn't", "aren't", "weren't", "hasn't", "haven't", "hadn't"}
STOPWORDS = set(stopwords.words("english")) - NEGATIONS


def clean_text(text):
    """
    Full pipeline: lowercase -> remove URLs/HTML/special chars ->
    tokenize -> remove stopwords (negations kept) -> lemmatize.
    Returns a single cleaned string, ready for TF-IDF in Phase 5.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)     # URLs
    text = re.sub(r"<.*?>", "", text)                # HTML tags
    text = re.sub(r"[^a-z\s]", "", text)              # keep letters + spaces only
    text = re.sub(r"\s+", " ", text).strip()          # collapse extra whitespace

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "ive made it through a week i just feel beaten and defeated",
        "i feel so worthless and weak what does he have",
        "im moved in ive been feeling kind of gloomy",
        "i am not happy about this at all",
    ]
    for s in samples:
        print(f"BEFORE: {s}")
        print(f"AFTER:  {clean_text(s)}\n")