# Emotion Detection from Text — Project Report

## Abstract

This project builds a machine learning system that classifies short text into one
of six emotions — sadness, joy, love, anger, fear, and surprise — using classical
NLP techniques. A TF-IDF + Linear SVM pipeline was selected after comparing three
candidate models, achieving 89.6% test accuracy and a macro F1-score of 0.849.
The system includes a full preprocessing pipeline, a persisted model for reuse,
and a prediction interface tested against both held-out test data and custom
real-world sentences.

## Introduction

Understanding the emotional tone of text has applications across customer
support, social media monitoring, and conversational AI. This project explores
a classical (non-deep-learning) approach to emotion classification, prioritizing
interpretability and a clear understanding of each pipeline stage over raw
model complexity.

## Problem Statement

Given a piece of text, predict the single emotion it most strongly expresses,
selecting from six categories: sadness, joy, love, anger, fear, surprise.

## Objectives

- Build a complete, reusable text preprocessing pipeline
- Compare multiple feature extraction and classification approaches empirically
- Evaluate using metrics appropriate for imbalanced data, not just accuracy
- Produce a prediction system with confidence scores
- Package the project for GitHub portfolio use and live deployment

## Dataset Description

**Source:** `dair-ai/emotion` (Hugging Face Datasets)
**Size:** 16,000 train / 2,000 validation / 2,000 test examples
**Columns:** `text` (raw tweet), `label` (integer, 0–5)

Class distribution (training set) revealed significant imbalance:

| Emotion | Count |
|---|---|
| joy | 5,362 |
| sadness | 4,666 |
| anger | 2,159 |
| fear | 1,937 |
| love | 1,304 |
| surprise | 572 |

![Class Distribution](figures/class_distribution.png)

Joy has roughly 9.4x more training examples than surprise — a factor that
shaped both the evaluation metric choice (below) and later error patterns.

## Methodology

The project follows a two-pipeline architecture: a **training pipeline** (run
once, produces a saved model and vectorizer) and an **inference pipeline**
(reused for every prediction, sharing the exact same preprocessing and
vectorizer as training to prevent train/inference mismatch).

## Data Preprocessing

Raw text was cleaned using the following steps, in this specific order:

1. Lowercasing
2. URL and HTML tag removal
3. Special character/number removal (letters and spaces only)
4. Tokenization
5. Stopword removal — **with negation words (`not`, `no`, `never`, etc.)
   deliberately excluded from the stopword list**, since removing them would
   invert sentence meaning (e.g. "not happy" → "happy")
6. Lemmatization

**Known limitation:** `WordNetLemmatizer` defaults to treating words as nouns
unless given explicit part-of-speech tags, so some verb forms (e.g. "beaten")
are not fully reduced to their root form.

## Feature Engineering

Three feature representations were considered:

| Method | Verdict |
|---|---|
| Bag-of-Words | Rejected — treats all words as equally important |
| **TF-IDF** | **Selected** — down-weights generic words, boosts distinctive ones |
| Word Embeddings | Deferred — added complexity not justified for a classical baseline |

TF-IDF was fit with `max_features=5000`, fit **only on training data** to
prevent data leakage; validation and test sets were transformed using the
already-fitted vectorizer.

**Concrete evidence TF-IDF worked as intended:** for the tweet *"didnt feel
humiliated"*, the word "feel" (the single most frequent word in the entire
corpus) received a low weight of 0.161, while "humiliated" (rare, emotionally
specific) received 0.787 — despite "feel" being far more common in the raw text.

## Model Development

Three models were trained and compared on the validation set:

| Model | Accuracy | Macro F1 | Train Time |
|---|---|---|---|
| Logistic Regression | 0.877 | 0.837 | 1.30s |
| Multinomial Naive Bayes | 0.747 | 0.571 | 0.30s |
| **Linear SVM** | **0.902** | **0.873** | 0.67s |

**Linear SVM was selected** — highest accuracy and macro F1, faster than
Logistic Regression. Naive Bayes' much larger accuracy–F1 gap (0.176 vs. ~0.03
for the others) indicated it was disproportionately failing on minority
classes, consistent with its unrealistic word-independence assumption.

Since `LinearSVC` does not natively output probabilities, the final model
wraps it in `CalibratedClassifierCV` (5-fold) to produce confidence scores
required by the prediction system.

## Evaluation Metrics

**Why macro F1 over accuracy:** with a 9.4x class imbalance, a model could
score well on accuracy while completely failing on minority classes. Macro
F1 averages per-class F1 scores equally, exposing this weakness.

## Results

**Test set performance:** Accuracy 89.6%, Macro F1 0.849

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| sadness | 0.93 | 0.93 | 0.93 | 581 |
| joy | 0.92 | 0.93 | 0.92 | 695 |
| love | 0.78 | 0.81 | 0.79 | 159 |
| anger | 0.88 | 0.89 | 0.88 | 275 |
| fear | 0.88 | 0.85 | 0.87 | 224 |
| surprise | 0.73 | 0.67 | 0.70 | 66 |

![Confusion Matrix](figures/confusion_matrix.png)

The two smallest classes (surprise, love) had the two lowest F1 scores,
directly tracing back to their limited training data. Two specific confusion
patterns emerged:

- **Love ↔ Joy** (33 joy tweets misclassified as love, 26 love tweets
  misclassified as joy) — both are positive-valence emotions with overlapping
  vocabulary.
- **Surprise → Fear** (12 of 66 surprise tweets misclassified as fear, the
  largest single error source for that class) — both are high-arousal,
  unexpected-event emotions with overlapping language.

**Custom sentence testing:** 7 of 10 hand-written sentences were classified
correctly. All 3 errors defaulted to "joy," the majority class — evidence that
under uncertainty, the model leans toward the largest class. Sentences using
direct emotion vocabulary ("I'm furious," "I'm scared") were classified with
high confidence (94%, 92%); sentences using indirect/idiomatic phrasing
("nothing seems to be going right") were classified with much lower confidence
or incorrectly — indicating a distribution shift between the dataset's short,
direct tweet style and natural conversational English.

## Challenges

- An environment-specific import hang (seaborn → scipy → NumPy's f2py module)
  during EDA, resolved by using plain matplotlib for simple plots
- Balancing preprocessing aggressiveness against meaning preservation
  (specifically, protecting negation words from stopword removal)

## Limitations

- Class imbalance measurably reduces performance on minority emotions
  (surprise, love) and biases ambiguous predictions toward the majority class
- TF-IDF has no concept of word order — "not happy" and a hypothetical
  "happy not" would vectorize identically outside of the specific negation
  words themselves being retained as separate features
- Lemmatization is not POS-aware, leaving some verb forms unreduced
- The training data's narrow stylistic range (short, direct tweets) limits
  generalization to more naturally-phrased real-world sentences

## Future Improvements

- Word embeddings or a transformer-based model (e.g. DistilBERT) to capture
  context and word order
- Class balancing techniques (oversampling minority classes, class-weighted
  loss) to address the imbalance directly rather than only accounting for it
  in evaluation
- POS-aware lemmatization for more accurate root-word reduction
- Training data augmentation with more naturally-phrased text to reduce
  distribution shift
- Live deployment via Streamlit for interactive, real-world testing

## Conclusion

This project delivered a complete, working emotion classification system with
strong overall performance (89.6% accuracy, 0.849 macro F1), while being
transparent about where and why it struggles. Choosing macro F1 as the primary
metric, protecting negation words during preprocessing, and testing against
custom sentences (not just the test set) surfaced concrete, explainable
limitations rather than an overstated success narrative — an approach that
strengthens rather than weakens the project's credibility.