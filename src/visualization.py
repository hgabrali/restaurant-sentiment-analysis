"""
Visualization Module for Restaurant Sentiment Analysis.

Provides charts for EDA, model evaluation, and business reporting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_COLORS = {
    "positive": "#10B981",
    "neutral": "#F59E0B",
    "negative": "#EF4444",
}


# ── EDA Plots ─────────────────────────────────────────────────────────────────

def plot_sentiment_distribution(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment",
    save: bool = True,
) -> plt.Figure:
    """Bar chart and pie chart of sentiment class distribution."""
    counts = df[sentiment_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = [SENTIMENT_COLORS.get(s, "#6B7280") for s in counts.index]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", alpha=0.9)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Sentiment Class Distribution (Count)", fontsize=14)
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Reviews")

    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Sentiment Distribution (%)", fontsize=14)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "sentiment_distribution.png", dpi=150, bbox_inches="tight")
    return fig


def plot_rating_distribution(
    df: pd.DataFrame,
    rating_col: str = "rating",
    sentiment_col: str = "sentiment",
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart of star ratings coloured by sentiment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rating_sentiment = df.groupby([rating_col, sentiment_col]).size().unstack(fill_value=0)
    rating_sentiment.plot(
        kind="bar",
        ax=ax,
        color=[SENTIMENT_COLORS.get(s, "#6B7280") for s in rating_sentiment.columns],
        edgecolor="white",
        alpha=0.9,
    )
    ax.set_title("Rating Distribution by Sentiment", fontsize=14)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title="Sentiment")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "rating_distribution.png", dpi=150, bbox_inches="tight")
    return fig


def plot_review_length_distribution(
    df: pd.DataFrame,
    length_col: str = "word_count",
    sentiment_col: str = "sentiment",
    save: bool = True,
) -> plt.Figure:
    """KDE plot of review word count by sentiment class."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for sentiment, group in df.groupby(sentiment_col):
        group[length_col].plot.kde(
            ax=ax,
            label=sentiment,
            color=SENTIMENT_COLORS.get(sentiment, "#6B7280"),
            linewidth=2,
        )
    ax.set_title("Review Length Distribution by Sentiment", fontsize=14)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.legend(title="Sentiment")
    ax.set_xlim(0, df[length_col].quantile(0.98))
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "review_length_distribution.png", dpi=150, bbox_inches="tight")
    return fig


def plot_wordcloud(
    df: pd.DataFrame,
    sentiment: str = "positive",
    text_col: str = "cleaned_text",
    save: bool = True,
) -> Optional[plt.Figure]:
    """Generate a word cloud for a given sentiment class."""
    if not WORDCLOUD_AVAILABLE:
        print("WordCloud not installed. Run: pip install wordcloud")
        return None

    subset = df[df["sentiment"] == sentiment][text_col].dropna()
    all_text = " ".join(subset.values)

    color = {"positive": "#10B981", "neutral": "#F59E0B", "negative": "#EF4444"}.get(sentiment, "#6B7280")
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Greens" if sentiment == "positive" else "Reds" if sentiment == "negative" else "Oranges",
        max_words=100,
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Most Common Words in {sentiment.capitalize()} Reviews", fontsize=16)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / f"wordcloud_{sentiment}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_top_terms(
    df: pd.DataFrame,
    sentiment: str = "positive",
    text_col: str = "cleaned_text",
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of top TF-IDF terms for a sentiment class."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    subset = df[df["sentiment"] == sentiment][text_col].dropna()
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = tfidf.fit_transform(subset)
    scores = np.asarray(X.mean(axis=0)).flatten()
    vocab = tfidf.get_feature_names_out()
    top_idx = scores.argsort()[-top_n:][::-1]
    top_terms = vocab[top_idx]
    top_scores = scores[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    color = SENTIMENT_COLORS.get(sentiment, "#6B7280")
    ax.barh(top_terms[::-1], top_scores[::-1], color=color, alpha=0.85, edgecolor="white")
    ax.set_title(f"Top {top_n} Terms — {sentiment.capitalize()} Reviews", fontsize=14)
    ax.set_xlabel("Mean TF-IDF Score")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / f"top_terms_{sentiment}.png", dpi=150, bbox_inches="tight")
    return fig


# ── Model Evaluation Plots ────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[List[str]] = None,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Plot a normalised confusion matrix."""
    if labels is None:
        labels = ["negative", "neutral", "positive"]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(" ", "_")
        fig.savefig(FIGURE_DIR / f"confusion_matrix_{fname}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    results: Dict[str, Dict],
    metric: str = "f1",
    save: bool = True,
) -> plt.Figure:
    """Bar chart comparing models on accuracy or F1."""
    rows = [
        {"Model": name, metric.upper(): res["metrics"][metric]}
        for name, res in results.items()
    ]
    df = pd.DataFrame(rows).sort_values(metric.upper(), ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("Blues_r", n_colors=len(df))
    bars = ax.bar(df["Model"], df[metric.upper()], color=colors, alpha=0.9, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title(f"Model Comparison — {metric.upper()}", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / f"model_comparison_{metric}.png", dpi=150, bbox_inches="tight")
    return fig


# ── Interactive Plotly Charts ─────────────────────────────────────────────────

def interactive_sentiment_over_time(
    df: pd.DataFrame,
    date_col: str = "review_date",
    sentiment_col: str = "sentiment",
) -> go.Figure:
    """Interactive line chart of sentiment trends over time."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").astype(str)
    trend = df.groupby(["month", sentiment_col]).size().reset_index(name="count")

    fig = px.line(
        trend,
        x="month",
        y="count",
        color=sentiment_col,
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Trend Over Time",
        labels={"month": "Month", "count": "Number of Reviews"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def interactive_restaurant_comparison(
    df: pd.DataFrame,
    restaurant_col: str = "restaurant_name",
    sentiment_col: str = "sentiment",
) -> go.Figure:
    """Interactive grouped bar chart comparing sentiment across restaurants."""
    comp = df.groupby([restaurant_col, sentiment_col]).size().reset_index(name="count")
    fig = px.bar(
        comp,
        x=restaurant_col,
        y="count",
        color=sentiment_col,
        color_discrete_map=SENTIMENT_COLORS,
        barmode="group",
        title="Sentiment Comparison Across Restaurants",
    )
    fig.update_layout(template="plotly_white")
    return fig


def interactive_review_explorer(df: pd.DataFrame) -> go.Figure:
    """Interactive scatter plot of review word count vs rating coloured by sentiment."""
    fig = px.scatter(
        df,
        x="word_count",
        y="rating",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hover_data=["review_text", "restaurant_name"] if "restaurant_name" in df.columns else None,
        title="Review Word Count vs Rating by Sentiment",
        opacity=0.6,
    )
    fig.update_layout(template="plotly_white")
    return fig
