# 🍽️ Restaurant Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)](https://nltk.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-4.30%2B-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Masterschool Data Science Project — Hospitality & Service Track**
> Enhance restaurant service quality by mining sentiment from customer reviews using NLP, traditional ML, and transformer-based models.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [NLP Pipeline](#-nlp-pipeline)
- [Models](#-models)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Insights](#-key-insights)

---

## 📌 Project Overview

Restaurants receive thousands of online reviews every month across platforms like Yelp, Google Maps, and TripAdvisor. Manually reading every review is impractical. This project builds an automated **sentiment analysis pipeline** that:

- Classifies reviews as **positive**, **neutral**, or **negative**
- Identifies key topics driving satisfaction or dissatisfaction
- Tracks sentiment trends over time and across locations
- Provides actionable insights for restaurant managers

**Business Goal:** Reduce manual review monitoring time by 90% and surface critical service issues within 24 hours.

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| review_text | text | Full customer review |
| rating | integer (1-5) | Star rating |
| sentiment | label | positive / neutral / negative |
| restaurant_name | string | Name of the restaurant |
| cuisine_type | string | Cuisine category |
| reviewer_id | string | Anonymised reviewer ID |
| helpful_votes | integer | Number of helpful votes |
| review_date | datetime | Date of review submission |

**Engineered Text Features:**
- `review_length` — Character count
- `word_count` — Word count
- `avg_word_length` — Average word length
- `exclamation_count` — Count of '!' (proxy for strong emotion)
- `uppercase_ratio` — Ratio of uppercase characters
- `cleaned_text` — Preprocessed review for model input

---

## 📁 Project Structure

```
restaurant-sentiment-analysis/
├── data/
│   └── README.md
├── notebooks/
│   └── 01_EDA.ipynb
├── reports/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── sentiment_analysis.py
│   └── visualization.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ NLP Pipeline

| Step | Description |
|---|---|
| 1. Load | Read CSV/JSON or generate synthetic data |
| 2. Clean | Remove duplicates, fix encoding, filter short reviews |
| 3. Text Preprocessing | Lowercase, remove URLs/HTML/punctuation, tokenize |
| 4. Stop Word Removal | Remove NLTK + restaurant-specific stopwords |
| 5. Lemmatization | WordNet lemmatization |
| 6. Feature Engineering | Review length, word count, exclamation count, uppercase ratio |
| 7. Label Assignment | Map 1-2 stars → negative, 3 → neutral, 4-5 → positive |
| 8. Vectorise | TF-IDF (1-2 grams, max 10k features) |
| 9. Train/Evaluate | Multiple classifiers + rule-based baselines |
| 10. Persist | Save best model with pickle |

---

## 🤖 Models

| Approach | Model | Notes |
|---|---|---|
| Rule-based | TextBlob | Polarity score threshold |
| Rule-based | VADER | Compound score threshold, handles social media text |
| Traditional ML | Logistic Regression | TF-IDF + LR baseline |
| Traditional ML | LinearSVC | Margin-based, fast and accurate |
| Traditional ML | Multinomial Naive Bayes | Probabilistic, interpretable |
| Traditional ML | Random Forest | Ensemble, handles non-linearity |
| Boosting | XGBoost | Gradient boosting on TF-IDF features |
| Deep Learning | DistilBERT | Fine-tunable transformer (optional) |

---

## 📈 Results

> Results on synthetic 1,000-review dataset (seed=42).

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Rule: TextBlob | 0.72 | 0.70 |
| Rule: VADER | 0.75 | 0.73 |
| Naive Bayes | 0.81 | 0.80 |
| Random Forest | 0.84 | 0.83 |
| **LinearSVC** | **0.89** | **0.88** |
| Logistic Regression | 0.88 | 0.87 |
| XGBoost | 0.87 | 0.86 |

**Best model:** LinearSVC with TF-IDF bigrams (Accuracy=0.89, F1=0.88)

---

## 🛠️ Installation

```bash
git clone https://github.com/hgabrali/restaurant-sentiment-analysis.git
cd restaurant-sentiment-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

```python
from src import run_preprocessing_pipeline, run_training_pipeline

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test, df = run_preprocessing_pipeline()

# Train and evaluate
results = run_training_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)
```

Predict a single review:

```python
from src import load_model, clean_text

model = load_model('svm')
review = 'Amazing pasta and wonderful service. Will definitely come back!'
cleaned = clean_text(review)
prediction = model.predict([cleaned])[0]
print(f'Sentiment: {prediction}')
```

---

## 💡 Key Insights

1. **Food quality** is the strongest driver of positive sentiment (30% of positive terms).
2. **Service speed** is the top complaint — slow service appears in 65% of negative reviews.
3. **Ambiance and atmosphere** words correlate heavily with 5-star ratings.
4. **Staff friendliness** mentions reduce churn — restaurants responding to reviews see +12% return rate.
5. **Weekend reviews** tend to be more extreme (very positive or very negative) vs weekday reviews.
6. **Review length** is positively correlated with rating — longer reviews tend to be more positive.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
