"""
sentiment_analysis.py
=====================
Model training, evaluation, persistence, and inference for
the Restaurant Sentiment Analysis project.

Supported models
----------------
- Rule-based  : TextBlob, VADER
- Traditional : Logistic Regression, LinearSVC, Multinomial Naive Bayes,
                Random Forest, XGBoost
                - Deep        : DistilBERT (optional, requires transformers + torch)
                """

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models")
LABEL_ORDER = ["negative", "neutral", "positive"]
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# TF-IDF default parameters
# ---------------------------------------------------------------------------
TFIDF_PARAMS: Dict[str, Any] = {
      "max_features": 10_000,
      "ngram_range": (1, 2),
      "sublinear_tf": True,
      "min_df": 2,
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_pipelines() -> Dict[str, Pipeline]:
      """Return a fresh dict of sklearn Pipelines keyed by model name."""
      return {
          "logistic_regression": Pipeline([
              ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
              ("clf", LogisticRegression(
                  max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
              )),
          ]),
          "svm": Pipeline([
              ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
              ("clf", CalibratedClassifierCV(
                  LinearSVC(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
              )),
          ]),
          "naive_bayes": Pipeline([
              ("tfidf", TfidfVectorizer(**TFIDF_PARAMS, use_idf=False)),
              ("clf", MultinomialNB(alpha=0.5)),
          ]),
          "random_forest": Pipeline([
              ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
              ("clf", RandomForestClassifier(
                  n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
              )),
          ]),
      }


# Optional XGBoost
def _try_add_xgboost(pipelines: Dict[str, Pipeline]) -> None:
      try:
                from xgboost import XGBClassifier  # type: ignore
        from sklearn.preprocessing import LabelEncoder

        class _XGBStringClassifier(XGBClassifier):
                      """Thin wrapper that encodes string labels for XGBoost."""
                      def __init__(self, **kwargs: Any) -> None:
                                        super().__init__(**kwargs)
                                        self._le = LabelEncoder()

                      def fit(self, X: Any, y: Any, **kw: Any) -> "_XGBStringClassifier":
                                        self._le.fit(y)
                                        return super().fit(X, self._le.transform(y), **kw)

                      def predict(self, X: Any) -> np.ndarray:
                                        return self._le.inverse_transform(super().predict(X))

                      def predict_proba(self, X: Any) -> np.ndarray:
                                        return super().predict_proba(X)

                  pipelines["xgboost"] = Pipeline([
                                ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
                                ("clf", _XGBStringClassifier(
                                                  n_estimators=300, learning_rate=0.1, max_depth=6,
                                                  use_label_encoder=False, eval_metric="mlogloss",
                                                  random_state=RANDOM_STATE,
                                )),
                  ])
except ImportError:
        pass


# ---------------------------------------------------------------------------
# Rule-based helpers
# ---------------------------------------------------------------------------

def _textblob_predict(texts: List[str]) -> np.ndarray:
      from textblob import TextBlob  # type: ignore

    preds = []
    for t in texts:
              score = TextBlob(t).sentiment.polarity
              if score > 0.1:
                            preds.append("positive")
elif score < -0.1:
            preds.append("negative")
else:
            preds.append("neutral")
      return np.array(preds)


def _vader_predict(texts: List[str]) -> np.ndarray:
      from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    import nltk
    try:
              nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()
    preds = []
    for t in texts:
              score = sia.polarity_scores(t)["compound"]
              if score >= 0.05:
                            preds.append("positive")
elif score <= -0.05:
            preds.append("negative")
else:
            preds.append("neutral")
      return np.array(preds)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
      """Print and return accuracy + weighted/macro F1."""
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (weighted)   : {f1_w:.4f}")
    print(f"  F1 (macro)      : {f1_m:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABEL_ORDER, zero_division=0))
    return {"model": model_name, "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, name: str, directory: Path = MODEL_DIR) -> Path:
      """Pickle *model* to *directory*/<name>.pkl and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pkl"
    with open(path, "wb") as f:
              pickle.dump(model, f)
          print(f"[INFO] Model saved → {path}")
    return path


def load_model(name: str, directory: Path = MODEL_DIR) -> Any:
      """Load a pickled model from *directory*/<name>.pkl.

          Parameters
              ----------
                  name : str
                          One of ``'logistic_regression'``, ``'svm'``, ``'naive_bayes'``,
                                  ``'random_forest'``, ``'xgboost'``.
                                      directory : Path
                                              Directory where models are stored (default: ``models/``).
                                                  """
    path = Path(directory) / f"{name}.pkl"
    if not path.exists():
              raise FileNotFoundError(
                            f"No saved model found at '{path}'. "
                            "Run run_training_pipeline() first to train and save models."
              )
          with open(path, "rb") as f:
                    return pickle.load(f)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
      train_df: pd.DataFrame,
      test_df: pd.DataFrame,
      text_col: str = "cleaned_text",
      label_col: str = "sentiment",
      save_models: bool = True,
      model_dir: Path = MODEL_DIR,
) -> pd.DataFrame:
      """Train all configured models, evaluate on *test_df*, and return a
          results DataFrame sorted by weighted F1.

              Parameters
                  ----------
                      train_df : pd.DataFrame
                              Training split (output of ``run_preprocessing_pipeline``).
                                  test_df : pd.DataFrame
                                          Test split.
                                              text_col : str
                                                      Column containing preprocessed review text.
                                                          label_col : str
                                                                  Column containing sentiment labels.
                                                                      save_models : bool
                                                                              Whether to pickle the best model to disk.
                                                                                  model_dir : Path
                                                                                          Where to save pickled models.

                                                                                              Returns
                                                                                                  -------
                                                                                                      pd.DataFrame
                                                                                                              Leaderboard sorted by ``f1_weighted`` descending.
                                                                                                                  """
    X_train = train_df[text_col].fillna("").tolist()
    y_train = train_df[label_col].tolist()
    X_test = test_df[text_col].fillna("").tolist()
    y_test = np.array(test_df[label_col].tolist())

    # Raw text for rule-based models
    raw_col = "review_text" if "review_text" in test_df.columns else text_col
    X_raw = test_df[raw_col].fillna("").tolist()

    pipelines = _build_pipelines()
    _try_add_xgboost(pipelines)

    results: List[Dict[str, float]] = []

    # --- sklearn pipelines ---
    for name, pipeline in pipelines.items():
              print(f"\n[TRAIN] {name} ...")
              pipeline.fit(X_train, y_train)
              y_pred = pipeline.predict(X_test)
              metrics = evaluate(y_test, y_pred, name)
              results.append(metrics)
              if save_models:
                            save_model(pipeline, name, model_dir)

          # --- Rule-based ---
          for name, fn in [("textblob", _textblob_predict), ("vader", _vader_predict)]:
                    try:
                                  print(f"\n[EVAL ] {name} (rule-based) ...")
                                  y_pred = fn(X_raw)
                                  metrics = evaluate(y_test, y_pred, name)
                                  results.append(metrics)
except ImportError as e:
            print(f"[WARN] Skipping {name}: {e}")

    leaderboard = (
              pd.DataFrame(results)
              .sort_values("f1_weighted", ascending=False)
              .reset_index(drop=True)
    )
    print("\n" + "="*50)
    print("LEADERBOARD")
    print("="*50)
    print(leaderboard.to_string(index=False))
    return leaderboard


# ---------------------------------------------------------------------------
# Single-review inference (convenience wrapper)
# ---------------------------------------------------------------------------

def predict_review(
      text: str,
      model_name: str = "logistic_regression",
      model_dir: Path = MODEL_DIR,
) -> str:
      """Predict sentiment for a single raw review string.

          Parameters
              ----------
                  text : str
                          Raw (unprocessed) review.
                              model_name : str
                                      Name of a trained model to load.
                                          model_dir : Path
                                                  Directory containing saved models.

                                                      Returns
                                                          -------
                                                              str
                                                                      One of ``'positive'``, ``'neutral'``, ``'negative'``.
                                                                          """
    from .data_preprocessing import clean_text  # local import to avoid circular
    model = load_model(model_name, model_dir)
    cleaned = clean_text(text)
    return model.predict([cleaned])[0]
