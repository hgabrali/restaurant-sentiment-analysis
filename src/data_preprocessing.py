"""
Data Preprocessing Module for Restaurant Sentiment Analysis.

Handles raw review data loading, text cleaning, tokenization,
feature engineering, and train/test splitting for NLP pipelines.
"""

import logging
import re
import string
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Download required NLTK resources
for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL = "sentiment"
TEXT_COL = "review_text"
RATING_COL = "rating"

SENTIMENT_MAP = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

RESTAURANT_STOPWORDS_EXTRA = {
    "restaurant", "food", "place", "came", "went", "go", "got", "get",
    "us", "also", "really", "would", "could", "even", "like", "one",
    "two", "three", "back", "time", "table", "order", "ordered",
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load restaurant review data from CSV or JSON file.

    Parameters
    ----------
    filepath : str
        Path to the data file (CSV or JSON).

    Returns
    -------
    pd.DataFrame with raw review data.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() == ".json":
        df = pd.read_json(filepath, lines=True)
    else:
        df = pd.read_csv(filepath)

    logger.info("Loaded %d reviews from %s", len(df), filepath)
    return df


def load_sample_data(n_reviews: int = 1000) -> pd.DataFrame:
    """Generate a synthetic restaurant review dataset for demonstration.

    Parameters
    ----------
    n_reviews : int
        Number of synthetic reviews to generate.

    Returns
    -------
    pd.DataFrame with synthetic restaurant reviews.
    """
    rng = np.random.default_rng(42)

    positive_phrases = [
        "Amazing food and excellent service! Will definitely come back.",
        "The pasta was absolutely delicious. Highly recommend!",
        "Fantastic atmosphere with very attentive staff. Loved every bite.",
        "Best pizza I have ever had. The crust was perfectly crispy.",
        "Outstanding dining experience. The chef clearly knows their craft.",
        "Wonderful flavors and generous portions. Great value for money.",
        "The steak was cooked to perfection. Impressive wine selection too.",
        "Lovely ambiance and the desserts are to die for.",
        "Friendly staff, quick service, and absolutely tasty food.",
        "A hidden gem! The risotto was creamy and full of flavor.",
    ]
    neutral_phrases = [
        "Decent place but nothing extraordinary. Food was okay.",
        "Average experience overall. Service was a bit slow.",
        "The food was acceptable but not memorable.",
        "Okay restaurant for a quick bite. Nothing special.",
        "Mixed feelings. Some dishes were good, others not so much.",
        "Standard quality. Would not go out of my way to visit again.",
        "Pretty typical for the price range. Nothing stood out.",
        "It was fine. The burger was decent but not impressive.",
    ]
    negative_phrases = [
        "Terrible service and bland food. Very disappointing.",
        "The wait was over an hour and the food was cold when it arrived.",
        "Rude staff and overpriced mediocre dishes. Will not return.",
        "Food tasted stale and the place was not clean.",
        "Worst dining experience ever. Everything went wrong.",
        "Extremely slow service and the portion sizes were tiny.",
        "The dish was completely different from what was described.",
        "Management needs serious improvement. Very unpleasant experience.",
    ]

    ratings = rng.choice([1, 2, 3, 4, 5], n_reviews, p=[0.10, 0.15, 0.15, 0.30, 0.30])
    reviews = []
    for r in ratings:
        if r >= 4:
            reviews.append(rng.choice(positive_phrases))
        elif r == 3:
            reviews.append(rng.choice(neutral_phrases))
        else:
            reviews.append(rng.choice(negative_phrases))

    restaurants = [
        "La Bella Italia", "The Golden Spoon", "Sakura Garden",
        "El Rancho", "The Rustic Table", "Spice Route",
    ]
    categories = ["Italian", "American", "Japanese", "Mexican", "Fusion", "Indian"]

    df = pd.DataFrame({
        TEXT_COL: reviews,
        RATING_COL: ratings,
        "restaurant_name": rng.choice(restaurants, n_reviews),
        "cuisine_type": rng.choice(categories, n_reviews),
        "reviewer_id": [f"user_{i:04d}" for i in rng.integers(1, 500, n_reviews)],
        "helpful_votes": rng.integers(0, 50, n_reviews),
        "review_date": pd.date_range("2022-01-01", periods=n_reviews, freq="6h"),
    })
    df[TARGET_COL] = df[RATING_COL].map(SENTIMENT_MAP)
    logger.info("Generated synthetic dataset with %d reviews", n_reviews)
    return df


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Clean and normalize a single review text.

    Steps
    -----
    1. Lowercase
    2. Remove URLs, HTML tags, and special characters
    3. Remove punctuation and digits
    4. Tokenize
    5. Remove stopwords (optional)
    6. Lemmatize (optional)

    Parameters
    ----------
    text : str
        Raw review text.
    remove_stopwords : bool
        Whether to remove English stopwords.
    lemmatize : bool
        Whether to apply WordNet lemmatization.

    Returns
    -------
    str: cleaned and normalized text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove emojis and non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    if remove_stopwords:
        stop = set(stopwords.words("english")) | RESTAURANT_STOPWORDS_EXTRA
        tokens = [t for t in tokens if t not in stop and len(t) > 2]

    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_texts(
    texts: pd.Series,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> pd.Series:
    """Apply clean_text() to a pandas Series of reviews.

    Parameters
    ----------
    texts : pd.Series
        Series of raw review strings.

    Returns
    -------
    pd.Series of cleaned texts.
    """
    logger.info("Cleaning %d review texts...", len(texts))
    cleaned = texts.apply(lambda t: clean_text(t, remove_stopwords, lemmatize))
    logger.info("Text cleaning complete.")
    return cleaned


# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived text and metadata features.

    New features
    ------------
    - review_length       : Character count of raw review
    - word_count          : Word count of raw review
    - avg_word_length     : Average word length
    - exclamation_count   : Number of '!' characters
    - question_count      : Number of '?' characters
    - uppercase_ratio     : Ratio of uppercase characters
    - cleaned_text        : Preprocessed review text

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TEXT_COL column.

    Returns
    -------
    pd.DataFrame with new feature columns.
    """
    df = df.copy()
    text = df[TEXT_COL].fillna("")

    df["review_length"] = text.str.len()
    df["word_count"] = text.str.split().str.len().fillna(0).astype(int)
    df["avg_word_length"] = text.apply(
        lambda t: np.mean([len(w) for w in str(t).split()]) if str(t).split() else 0
    )
    df["exclamation_count"] = text.str.count("!")
    df["question_count"] = text.str.count("\?")
    df["uppercase_ratio"] = text.apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df["cleaned_text"] = preprocess_texts(text)

    logger.info("Feature engineering complete. Total columns: %d", df.shape[1])
    return df


# ── Data Cleaning ──────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle missing values, and standardise columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Cleaned pd.DataFrame.
    """
    df = df.copy()
    before = len(df)
    df.drop_duplicates(subset=[TEXT_COL], inplace=True)
    df.dropna(subset=[TEXT_COL, TARGET_COL], inplace=True)
    df = df[df[TEXT_COL].str.strip().str.len() > 5]
    logger.info("Removed %d rows during cleaning. Remaining: %d", before - len(df), len(df))
    return df


# ── Train/Test Split ──────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    feature_col: str = "cleaned_text",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple:
    """Split into train, validation, and test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[feature_col]
    y = df[TARGET_COL]

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=random_state, stratify=y_tv
    )
    logger.info(
        "Split — train: %d | val: %d | test: %d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(filepath: Optional[str] = None) -> Tuple:
    """Execute the complete preprocessing pipeline.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV/JSON data. If None, synthetic data is used.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, df_processed
    """
    df = load_data(filepath) if filepath else load_sample_data()
    df = clean_dataset(df)
    df = engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    logger.info("Preprocessing pipeline complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test, df


if __name__ == "__main__":
    X_tr, X_v, X_te, y_tr, y_v, y_te, df = run_preprocessing_pipeline()
    print(f"Train: {len(X_tr)} | Val: {len(X_v)} | Test: {len(X_te)}")
    print(f"Label distribution:\n{y_tr.value_counts()}")
