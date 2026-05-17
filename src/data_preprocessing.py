"""
  data_preprocessing.py
  =====================
  Text cleaning, feature engineering, and train/val/test split pipeline
    for the Restaurant Sentiment Analysis project.
    """

    from __future__ import annotations

    import re
    import string
    import pickle
    from pathlib import Path
    from typing import Tuple

    import numpy as np
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from sklearn.model_selection import train_test_split

    # ---------------------------------------------------------------------------
    # NLTK resource bootstrap
    # ---------------------------------------------------------------------------
    _NLTK_RESOURCES = [
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"
    ]

    def _ensure_nltk_resources() -> None:
    for resource in _NLTK_RESOURCES:
              try:
                  nltk.data.find(f"tokenizers/{resource}")
              except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
                              except LookupError:
                nltk.download(resource, quiet=True)

                  _ensure_nltk_resources()

                  # ---------------------------------------------------------------------------
                  # Constants
                  # ---------------------------------------------------------------------------
                  RESTAURANT_STOPWORDS: set[str] = {
                        "restaurant", "food", "place", "time", "came", "went", "got", "get",
                        "also", "would", "go", "back", "really", "like", "one", "even",
                        "us", "table", "order", "ordered", "menu", "eat", "eating",
                  }

LABEL_MAP: dict[int, str] = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}

RANDOM_STATE: int = 42

  # ---------------------------------------------------------------------------
  # Text cleaning helpers
  # ---------------------------------------------------------------------------
  _lemmatizer = WordNetLemmatizer()
  _stop_words: set[str] | None = None


    def _get_stop_words() -> set[str]:
    global _stop_words
          if _stop_words is None:
              _stop_words = set(stopwords.words("english")) | RESTAURANT_STOPWORDS
          return _stop_words


      def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS treebank tag to WordNet POS constant."""
          if treebank_tag.startswith("J"):
              return wordnet.ADJ
          if treebank_tag.startswith("V"):
              return wordnet.VERB
          if treebank_tag.startswith("R"):
              return wordnet.ADV
          return wordnet.NOUN


      def clean_text(text: str) -> str:
    """Full NLP preprocessing pipeline for a single review string.

          Steps: lowercase → strip URLs/HTML → remove punctuation/digits →
                tokenise → remove stopwords → lemmatise.

                Parameters
                ----------
                text : str
                          Raw review text.

                      Returns
                      -------
                      str
                          Space-joined lemmatised tokens, ready for vectorisation.
                                """
                                if not isinstance(text, str) or not text.strip():
                                    return ""

                                text = text.lower()
                                text = re.sub(r"https?://\S+|www\.\S+", " ", text)
                                text = re.sub(r"<[^>]+>", " ", text)
                                text = re.sub(r"[^a-z\s]", " ", text)
                                text = re.sub(r"\s+", " ", text).strip()

                                tokens = word_tokenize(text)
                                stop_words = _get_stop_words()
                                tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

                                pos_tags = nltk.pos_tag(tokens)
                                tokens = [_lemmatizer.lemmatize(tok, _get_wordnet_pos(pos)) for tok, pos in pos_tags]

                                return " ".join(tokens)


                            # ---------------------------------------------------------------------------
                            # Feature engineering
                            # ---------------------------------------------------------------------------

                            def engineer_features(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
                                """Add hand-crafted text features to *df* in-place and return it."""
                                df = df.copy()
                                raw = df[text_col].fillna("")
                                df["review_length"] = raw.str.len()
                                df["word_count"] = raw.str.split().str.len()
                                df["avg_word_length"] = raw.apply(
                                    lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0.0
                                )
                                df["exclamation_count"] = raw.str.count(r"!")
                                df["question_count"] = raw.str.count(r"\?")
                                df["uppercase_ratio"] = raw.apply(
                                    lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
                                )
                                df["cleaned_text"] = raw.apply(clean_text)
                                return df


                            # ---------------------------------------------------------------------------
                            # Label assignment
                            # ---------------------------------------------------------------------------

                            def assign_labels(df: pd.DataFrame, rating_col: str = "rating") -> pd.DataFrame:
    """Map integer star ratings (1-5) to sentiment labels and return df."""
          df = df.copy()
          df["sentiment"] = df[rating_col].map(LABEL_MAP)
          return df


      # ---------------------------------------------------------------------------
      # Synthetic data generator (fallback when no CSV is provided)
      # ---------------------------------------------------------------------------

      def generate_synthetic_data(n_samples: int = 1000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic restaurant review dataset for testing."""
          rng = np.random.default_rng(seed)

          positive_phrases = [
              "Absolutely amazing food and wonderful service",
              "Best dining experience I have ever had",
              "Great atmosphere friendly staff and delicious meals",
              "Outstanding quality will definitely come back",
              "Fantastic flavours and very attentive waiters",
          ]
          neutral_phrases = [
              "Decent place nothing special but ok",
              "Average food and standard service",
              "It was fine not great not terrible",
              "Pretty ordinary experience overall",
              "Neither impressed nor disappointed",
          ]
          negative_phrases = [
              "Terrible food and rude staff awful experience",
              "Waited forever and the meal was cold",
              "Completely wrong order and no apology",
              "Would never return truly disappointing",
              "Very overpriced for such poor quality",
          ]

          phrase_pools = {5: positive_phrases, 4: positive_phrases,
                                              3: neutral_phrases, 2: negative_phrases, 1: negative_phrases}

    cuisines = ["Italian", "Mexican", "Chinese", "Indian", "Japanese", "American"]
          restaurants = ["The Golden Fork", "Spice Garden", "Bamboo House",
                                            "Pasta Palace", "Taco Town", "Sushi World"]

          records = []
          for i in range(n_samples):
              rating = int(rng.integers(1, 6))
              phrase = phrase_pools[rating][int(rng.integers(0, len(phrase_pools[rating])))]
              extras = ["with my family", "on a Friday night", "for a business lunch",
                                          "as a birthday treat", "after a long day"]
              review = f"{phrase} {extras[int(rng.integers(0, len(extras)))]}"
              records.append({
                            "review_text": review,
                            "rating": rating,
                            "restaurant_name": restaurants[int(rng.integers(0, len(restaurants)))],
                            "cuisine_type": cuisines[int(rng.integers(0, len(cuisines)))],
                            "reviewer_id": f"user_{i:04d}",
                            "helpful_votes": int(rng.integers(0, 20)),
                            "review_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
              })

          df = pd.DataFrame(records)
          return df


      # ---------------------------------------------------------------------------
      # Main pipeline
      # ---------------------------------------------------------------------------

      def run_preprocessing_pipeline(
          csv_path: str | Path | None = None,
          test_size: float = 0.15,
          val_size: float = 0.15,
          seed: int = RANDOM_STATE,
      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
          """Load (or generate) data, clean, feature-engineer, and split.

          Parameters
          ----------
          csv_path : str or Path, optional
                    Path to a labelled CSV with at least ``review_text`` and ``rating``
                    columns. If *None*, synthetic data is generated.
                test_size : float
                          Fraction of data reserved for the test set.
                      val_size : float
                          Fraction of data reserved for the validation set.
                      seed : int
                          Random seed for reproducibility.

                      Returns
                      -------
                      train_df, val_df, test_df, full_df : pd.DataFrame
                            """
                            # 1. Load or generate
                            if csv_path is not None:
                                df = pd.read_csv(csv_path)
                            else:
                                print("[INFO] No CSV provided — generating synthetic dataset (n=1000).")
                                df = generate_synthetic_data(seed=seed)

                            # 2. Basic cleaning
                            df = df.drop_duplicates(subset=["review_text"])
                            df = df[df["review_text"].str.split().str.len() >= 3].copy()
                            df = df.reset_index(drop=True)

                            # 3. Labels
                            if "sentiment" not in df.columns:
                                df = assign_labels(df)
                            df = df.dropna(subset=["sentiment"])

                            # 4. Feature engineering
                            df = engineer_features(df)

                            print(f"[INFO] Dataset size after cleaning: {len(df)} reviews")
                            print(f"[INFO] Sentiment distribution:\n{df['sentiment'].value_counts()}")

                            # 5. Split
                            train_val, test_df = train_test_split(
                                df, test_size=test_size, random_state=seed, stratify=df["sentiment"]
                            )
                            relative_val = val_size / (1 - test_size)
                            train_df, val_df = train_test_split(
                                      train_val, test_size=relative_val, random_state=seed, stratify=train_val["sentiment"]
                                  )

                                  print(
                                      f"[INFO] Split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
                                  )
                                  return train_df, val_df, test_df, df
