# 🍽️ Restaurant Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)](https://nltk.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-4.30%2B-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Masterschool Data Science Project — Hospitality & Service Track**
> > Enhance restaurant service quality by mining sentiment from customer reviews using NLP, traditional ML, and transformer-based models.
> >
> > **Version:** 2.0.0 | **Author:** Hande Gabrali-Knobloch | **Notebook:** [Colab](https://colab.research.google.com/drive/1DXn51z4XP4cVBx6h-3D797BfDDTUzaTk)
> >
> > ---
> >
> > ## 📋 Table of Contents
> >
> > - [Project Overview](#-project-overview)
> > - - [Dataset](#-dataset)
> >   - - [Project Structure](#-project-structure)
> >     - - [NLP Pipeline](#️-nlp-pipeline)
> >       - - [Models](#-models)
> >         - - [Results](#-results)
> >           - - [Installation](#️-installation)
> >             - - [Usage](#-usage)
> >               - - [Key Insights](#-key-insights)
> >                 - - [Limitations](#️-critique--limitations)
> >                  
> >                   - ---
> >
> > ## 📌 Project Overview
> >
> > Restaurants receive thousands of online reviews every month across platforms like Yelp, Google Maps, and TripAdvisor. Manually reading every review is impractical. This project builds an automated sentiment analysis pipeline that:
> >
> > - Classifies reviews as **positive**, **neutral**, or **negative**
> > - - Identifies key topics driving satisfaction or dissatisfaction
> >   - - Tracks sentiment trends over time and across locations
> >     - - Provides actionable insights for restaurant managers
> >      
> >       - **Business Goal:** Reduce manual review monitoring time by 90% and surface critical service issues within 24 hours.
> >      
> >       - ---
> >
> > ## 📊 Dataset
> >
> > | Feature | Type | Description |
> > |---|---|---|
> > | `review_text` | text | Full customer review |
> > | `rating` | integer (1–5) | Star rating |
> > | `sentiment` | label | positive / neutral / negative |
> > | `restaurant_name` | string | Name of the restaurant |
> > | `cuisine_type` | string | Cuisine category |
> > | `reviewer_id` | string | Anonymised reviewer ID |
> > | `helpful_votes` | integer | Number of helpful votes |
> > | `review_date` | datetime | Date of review submission |
> >
> > **Engineered Text Features:**
> >
> > - `review_length` — Character count
> > - - `word_count` — Word count
> >   - - `avg_word_length` — Average word length
> >     - - `exclamation_count` — Count of `!` (proxy for strong emotion)
> >       - - `question_count` — Count of `?` (complaint/inquiry proxy)
> >         - - `uppercase_ratio` — Ratio of uppercase characters
> >           - - `cleaned_text` — Preprocessed review for model input
> >            
> >             - **Label Assignment:**
> >            
> >             - | Star Rating | Sentiment Label |
> >             - |---|---|
> >             - | 1 – 2 stars | negative |
> > | 3 stars | neutral |
> > | 4 – 5 stars | positive |
> >
> > ---
> >
> > ## 📁 Project Structure
> >
> > ```
> > restaurant-sentiment-analysis/
> > ├── data/
> > │   └── README.md
> > ├── notebooks/
> > │   └── restaurant_sentiment_analysis_Masterschool_2026.ipynb
> > ├── reports/
> > │   └── README.md
> > ├── src/
> > │   ├── __init__.py
> > │   ├── data_preprocessing.py
> > │   ├── sentiment_analysis.py
> > │   └── visualization.py
> > ├── .gitignore
> > ├── LICENSE
> > ├── README.md
> > └── requirements.txt
> > ```
> >
> > ---
> >
> > ## ⚙️ NLP Pipeline
> >
> > | Step | Description |
> > |---|---|
> > | 1. Load | Read CSV/JSON or generate synthetic data |
> > | 2. Clean | Remove duplicates, fix encoding, filter short reviews |
> > | 3. Text Preprocessing | Lowercase, remove URLs/HTML/punctuation, tokenize |
> > | 4. Stop Word Removal | Remove NLTK + restaurant-specific stopwords |
> > | 5. Lemmatization | WordNet lemmatization |
> > | 6. Feature Engineering | review_length, word_count, exclamation_count, question_count, uppercase_ratio |
> > | 7. Label Assignment | Map 1–2 stars → negative, 3 → neutral, 4–5 → positive |
> > | 8. Vectorise | TF-IDF (1–2 grams, max 10 k features) |
> > | 9. Train/Evaluate | Multiple classifiers + rule-based baselines |
> > | 10. Persist | Save best model with pickle |
> >
> > ---
> >
> > ## 🤖 Models
> >
> > | Approach | Model | Notes | 
> > |---|---|---|
> > | Rule-based | TextBlob | Polarity score threshold |
> > | Rule-based | VADER | Compound score threshold, handles social media text |
> > | Traditional ML | Logistic Regression | TF-IDF + LR baseline |
> > | Traditional ML | LinearSVC | Margin-based, fast and accurate |
> > | Traditional ML | Multinomial Naive Bayes | Probabilistic, interpretable |
> > | Traditional ML | Random Forest | Ensemble, handles non-linearity |
> > | Boosting | XGBoost | Gradient boosting on TF-IDF features (XGBStringClassifier wrapper) |
> > | Deep Learning | DistilBERT | Fine-tunable transformer (optional) |
> >
> > ---
> >
> > ## 📈 Results
> >
> > Results on **synthetic 1,000-review dataset** (seed=42, 530 reviews after deduplication and cleaning).
> >
> > > ⚠️ **Important:** Near-perfect ML scores are an **artifact of the synthetic data** — the phrase pools used to generate reviews have no lexical overlap between classes, making them linearly separable. On real restaurant reviews expect roughly **0.75–0.90 weighted F1**.
> > >
> > > | Model | Accuracy | F1 (weighted) |
> > > |---|---|---|
> > > | Logistic Regression | 1.0000 | 1.0000 |
> > > | LinearSVC | 1.0000 | 1.0000 |
> > > | Naive Bayes | 1.0000 | 1.0000 |
> > > | Random Forest | 1.0000 | 1.0000 |
> > > | XGBoost | 0.9906 | 0.9905 |
> > > | Rule: TextBlob | 0.9151 | 0.9092 |
> > > | Rule: VADER | 0.8396 | 0.7858 |
> > >
> > > **Best model on synthetic data:** Logistic Regression / LinearSVC / Naive Bayes / Random Forest (all tied at F1 = 1.000)
> > > **Best rule-based:** TextBlob (Accuracy = 0.92, F1 = 0.91)
> > >
> > > ---
> > >
> > > ## 🛠️ Installation
> > >
> > > ```bash
> > > git clone https://github.com/hgabrali/restaurant-sentiment-analysis.git
> > > cd restaurant-sentiment-analysis
> > > python -m venv .venv
> > > source .venv/bin/activate        # Windows: .venv\Scripts\activate
> > > pip install -r requirements.txt
> > > python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4 vader_lexicon
> > > ```
> > >
> > > ---
> > >
> > > ## 🚀 Usage
> > >
> > > ```python
> > > from src import run_preprocessing_pipeline, run_training_pipeline
> > >
> > > # Preprocess — returns DataFrames, not raw arrays
> > > train_df, val_df, test_df, full_df = run_preprocessing_pipeline()
> > >
> > > # Train and evaluate all models
> > > results = run_training_pipeline(train_df, test_df)
> > > ```
> > >
> > > Predict a single review:
> > >
> > > ```python
> > > from src import load_model, clean_text
> > >
> > > model = load_model('logistic_regression')   # or 'svm', 'naive_bayes', etc.
> > > review = 'Amazing pasta and wonderful service. Will definitely come back!'
> > > cleaned = clean_text(review)
> > > prediction = model.predict([cleaned])[0]
> > > print(f'Sentiment: {prediction}')
> > > ```
> > >
> > > ---
> > >
> > > ## 💡 Key Insights
> > >
> > > > All insights below are **computed from the actual dataset** via the notebook — not assumed in advance.
> > > >
> > > > 1. **Dataset:** 530 cleaned reviews across **6 restaurants** / **6 cuisines** (after deduplication from 1,000 synthetic reviews).
> > > > 2. 2. **Class balance:** negative = 142 (26.8%) | neutral = 84 (15.8%) | positive = 304 (57.4%)
> > > >    3.    — `neutral` is the minority class; use **macro-F1** as the headline metric on real data.
> > > >    4.3. **Mean word count by sentiment:** negative ≈ 24 | neutral ≈ 26 | positive ≈ 24
> > > >         — Neutral reviews are marginally longer on average (mixed reviews carry more detail).
> > > >      4. **Top positive terms (TF-IDF):** `every`, `experience`, `great`, `friend`, `wait`
> > > >      5. 5. **Top negative terms (TF-IDF):** `experience`, `wrong`, `almost`, `returning`, `certainly`
> > > >         6. 6. **Word-count vs rating correlation:** r = −0.170 (weak negative — longer reviews lean slightly negative)
> > > >            7. 7. **Rating by cuisine:** best = Italian (avg 3.57 ★) | worst = Indian (avg 3.40 ★)
> > > >               8. 8. **Rule-based baselines:** TextBlob (F1 = 0.91) outperforms VADER (F1 = 0.79) on this synthetic corpus; results may differ on real noisy text.
> > > >                  9. 9. **XGBoost** is the only ML model below perfect on synthetic data (F1 = 0.990), reflecting its sensitivity to class imbalance and token sparsity.
> > > >                    
> > > >                     10. ---
> > > >                    
> > > >                     11. ## ⚠️ Critique & Limitations
> > > >                    
> > > >                     12. ### Methodology
> > > >
> > > > - **Perfect synthetic scores are an artifact.** The synthetic generator builds each review from sentiment-specific phrase pools with no lexical overlap — every ML model reaches ≈1.00 F1. Upload a labelled CSV before drawing any conclusion about real-world model quality.
> > > > - - **Rating → sentiment mapping loses signal.** Folding 1–2★ into *negative* and 4–5★ into *positive* discards intensity, and treating the 3★ neutral band as a true class mixes genuine ambivalence with noise.
> > > >   - - **Class imbalance.** `neutral` is the minority class (15.8%). `class_weight="balanced"` helps the linear models, but **macro-F1 — not weighted F1 — is the honest headline metric**.
> > > >     - - **Correlation ≠ causation.** Rating correlations (§ Correlation Analysis) are descriptive; they are confounded by reviewer and cuisine effects not controlled for here.
> > > >      
> > > >       - ### Failure points in noisy data
> > > >      
> > > >       - - Cleaning strips digits, punctuation and emoji — sarcasm, negation ("not bad") and emoji sentiment are lost for ML models; only the rule-based path sees raw text.
> > > >         - - `LinearSVC` exposes no `predict_proba`; wrap with `CalibratedClassifierCV` for probability thresholding.
> > > >           - - TF-IDF + bigrams ignore word order beyond two tokens; genuinely ambiguous reviews need a contextual model (DistilBERT fine-tune).
> > > >            
> > > >             - ### Suggested next steps
> > > >            
> > > >             - - Add **macro-F1** and per-class PR curves to the leaderboard.
> > > >               - - Run `GridSearchCV` on `val_df` for hyperparameter tuning (`C`, `max_features`, `ngram_range`).
> > > >                 - - Benchmark a **DistilBERT fine-tune** against the classical models on a real review CSV.
> > > >                   - - Add `CalibratedClassifierCV` wrapper around `LinearSVC` for confidence scores.
> > > >                    
> > > >                     - ---
> > > >
> > > > ## 📄 License
> > > >
> > > > This project is licensed under the [MIT License](LICENSE).
