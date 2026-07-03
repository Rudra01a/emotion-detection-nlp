# Emotion Detection from Text (NLP)

A machine learning system that classifies text into emotions (joy, sadness, anger, fear, love, surprise) using classical NLP techniques, deployed as a live Streamlit app.

## Project structure

```
emotion-detection-nlp/
├── data/
│   ├── raw/          # original dataset, untouched
│   └── processed/    # cleaned/preprocessed data
├── src/
│   ├── preprocessing.py   # text cleaning, shared by training + inference
│   ├── train.py            # loads data, fits TF-IDF, trains model
│   ├── evaluate.py         # metrics + confusion matrix
│   └── predict.py          # predict_emotion(text) -> (label, confidence)
├── models/              # saved trained model + vectorizer (.pkl)
├── reports/figures/     # saved evaluation plots
├── app.py                # Streamlit live demo
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/train.py       # trains and saves model
python src/evaluate.py    # generates metrics + plots
streamlit run app.py      # launch local demo
```

## Status

In progress — Phase 1 (environment + Git/GitHub setup) complete.

## Approach

Baseline: TF-IDF + Logistic Regression, trained on the dair-ai/emotion dataset.
Evaluated with macro F1-score and confusion matrix (chosen over raw accuracy due to class imbalance).

## Live demo

Link added once deployed to Streamlit Community Cloud.

## Results

To be added after model evaluation.
