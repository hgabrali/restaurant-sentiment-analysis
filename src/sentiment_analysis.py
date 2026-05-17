"""
Sentiment Analysis Module for Restaurant Reviews.

Implements multiple approaches:
1. Rule-based: TextBlob and VADER
2. Traditional ML: TF-IDF + Logistic Regression / SVM / Random Forest / XGBoost
3. Deep Learning: Fine-tuned BERT / DistilBERT (via HuggingFace Transformers)
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from textblob import TextBlob

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}

# ── Evaluation Helper ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, average: str = "weighted") -> Dict[str, float]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    average : str
        Averaging strategy for F1 ('weighted', 'macro', 'micro').

    Returns
    -------
    dict with accuracy, f1, and full classification report string.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "f1": f1, "report": report}


def evaluate_model(
    model: Any,
    X: Any,
    y: Any,
    split_name: str = "test",
) -> Dict[str, float]:
    """Predict and evaluate a fitted model.

    Parameters
    ----------
    model : fitted estimator with predict() method.
    X : feature matrix or list of texts.
    y : true labels.
    split_name : str
        Label for logging.

    Returns
    -------
    dict with evaluation metrics.
    """
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred)
    logger.info(
        "[%s] Accuracy=%.4f | F1=%.4f", split_name, metrics["accuracy"], metrics["f1"]
    )
    logger.info("\n%s", metrics["report"])
    return metrics


# ── Rule-Based Approaches ─────────────────────────────────────────────────────

def textblob_sentiment(text: str) -> str:
    """Classify sentiment using TextBlob polarity score.

    Parameters
    ----------
    text : str
        Review text.

    Returns
    -------
    str: 'positive', 'neutral', or 'negative'
    """
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    return "neutral"


def vader_sentiment(text: str) -> str:
    """Classify sentiment using VADER compound score.

    Parameters
    ----------
    text : str
        Review text.

    Returns
    -------
    str: 'positive', 'neutral', or 'negative'
    """
    if not VADER_AVAILABLE:
        raise ImportError("VADER not available. Install nltk and run nltk.download('vader_lexicon')")
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def apply_rule_based(texts: pd.Series, method: str = "textblob") -> pd.Series:
    """Apply a rule-based sentiment method to a series of texts.

    Parameters
    ----------
    texts : pd.Series
    method : str
        'textblob' or 'vader'.

    Returns
    -------
    pd.Series of sentiment labels.
    """
    if method == "textblob":
        return texts.apply(textblob_sentiment)
    elif method == "vader":
        return texts.apply(vader_sentiment)
    raise ValueError(f"Unknown method: {method}. Choose 'textblob' or 'vader'.")


# ── TF-IDF Vectorisation ──────────────────────────────────────────────────────

def build_tfidf_vectorizer(
    max_features: int = 10_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Create a TF-IDF vectoriser with recommended settings.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple
        Range of n-grams to extract.
    min_df : int
        Minimum document frequency for a token.
    sublinear_tf : bool
        Apply sublinear TF scaling.

    Returns
    -------
    TfidfVectorizer (unfitted).
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
    )


# ── Traditional ML Models ─────────────────────────────────────────────────────

def build_logistic_regression_pipeline(
    max_features: int = 10_000,
    C: float = 1.0,
) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    max_features : int
        TF-IDF vocabulary size.
    C : float
        Inverse regularisation strength.

    Returns
    -------
    sklearn Pipeline (unfitted).
    """
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer(max_features=max_features)),
        ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def build_svm_pipeline(max_features: int = 10_000, C: float = 1.0) -> Pipeline:
    """Build a TF-IDF + LinearSVC pipeline."""
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer(max_features=max_features)),
        ("clf", LinearSVC(C=C, max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def build_naive_bayes_pipeline(max_features: int = 10_000) -> Pipeline:
    """Build a TF-IDF + Multinomial Naive Bayes pipeline."""
    from sklearn.preprocessing import FunctionTransformer
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)),
        ("to_abs", FunctionTransformer(np.abs)),
        ("clf", MultinomialNB()),
    ])


def build_random_forest_pipeline(
    max_features: int = 5_000,
    n_estimators: int = 200,
) -> Pipeline:
    """Build a TF-IDF + Random Forest pipeline."""
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer(max_features=max_features)),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )),
    ])


def build_xgboost_pipeline(max_features: int = 5_000) -> Pipeline:
    """Build a TF-IDF + XGBoost pipeline."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer(max_features=max_features)),
        ("clf", xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42,
        )),
    ])


def train_model(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    model_name: str = "model",
) -> Pipeline:
    """Fit a pipeline on training data and log results.

    Parameters
    ----------
    pipeline : sklearn Pipeline.
    X_train, y_train : training data.
    model_name : str
        Name for logging.

    Returns
    -------
    Fitted pipeline.
    """
    logger.info("Training %s...", model_name)
    pipeline.fit(X_train, y_train)
    logger.info("%s training complete.", model_name)
    return pipeline


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_model(model: Any, name: str, directory: str = "models") -> str:
    """Persist a model to disk."""
    path = Path(directory) / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved to %s", path)
    return str(path)


def load_model(name: str, directory: str = "models") -> Any:
    """Load a persisted model from disk."""
    path = Path(directory) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model


# ── Full Training Pipeline ────────────────────────────────────────────────────

def run_training_pipeline(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Train all models and evaluate on the test set.

    Parameters
    ----------
    X_train, y_train : training split.
    X_val, y_val : validation split.
    X_test, y_test : test split.

    Returns
    -------
    results : dict mapping model names to (model, metrics).
    """
    models_to_train = {
        "logistic_regression": build_logistic_regression_pipeline(),
        "svm": build_svm_pipeline(),
        "naive_bayes": build_naive_bayes_pipeline(),
        "random_forest": build_random_forest_pipeline(),
    }
    if XGB_AVAILABLE:
        models_to_train["xgboost"] = build_xgboost_pipeline()

    results = {}
    for name, pipeline in models_to_train.items():
        fitted = train_model(pipeline, X_train, y_train, model_name=name)
        metrics = evaluate_model(fitted, X_test, y_test, split_name=name)
        results[name] = {"model": fitted, "metrics": metrics}

    # Rule-based baselines
    for method in ["textblob", "vader"]:
        try:
            preds = apply_rule_based(X_test, method=method)
            metrics = compute_metrics(y_test, preds)
            results[f"rule_{method}"] = {"model": None, "metrics": metrics}
            logger.info("[rule_%s] Accuracy=%.4f | F1=%.4f", method, metrics["accuracy"], metrics["f1"])
        except Exception as e:
            logger.warning("Rule-based %s failed: %s", method, e)

    # Print leaderboard
    print("\n===== Sentiment Analysis Leaderboard (Test Set) =====")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 (weighted)':>15}")
    print("-" * 52)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["metrics"]["f1"]):
        m = res["metrics"]
        print(f"{name:<25} {m['accuracy']:>10.4f} {m['f1']:>15.4f}")

    return results


if __name__ == "__main__":
    from data_preprocessing import run_preprocessing_pipeline

    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_preprocessing_pipeline()
    results = run_training_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)

    best = max(results, key=lambda k: results[k]["metrics"]["f1"])
    print(f"\nBest model: {best} (F1={results[best]['metrics']['f1']:.4f})")
    if results[best]["model"] is not None:
        save_model(results[best]["model"], best)
