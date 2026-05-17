"""
src — Restaurant Sentiment Analysis Package.

Modules
-------
data_preprocessing : Data loading, text cleaning, feature engineering.
sentiment_analysis : Rule-based, ML, and deep learning sentiment models.
visualization      : EDA and model evaluation plotting utilities.
"""

__version__ = "1.0.0"
__author__ = "Hande Gabrali-Knobloch"
__license__ = "MIT"

from .data_preprocessing import (
    clean_dataset,
    clean_text,
    engineer_features,
    load_data,
    load_sample_data,
    preprocess_texts,
    run_preprocessing_pipeline,
    split_dataset,
)
from .sentiment_analysis import (
    apply_rule_based,
    build_logistic_regression_pipeline,
    build_naive_bayes_pipeline,
    build_random_forest_pipeline,
    build_svm_pipeline,
    build_tfidf_vectorizer,
    compute_metrics,
    evaluate_model,
    load_model,
    run_training_pipeline,
    save_model,
    textblob_sentiment,
    train_model,
    vader_sentiment,
)
from .visualization import (
    interactive_restaurant_comparison,
    interactive_review_explorer,
    interactive_sentiment_over_time,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_rating_distribution,
    plot_review_length_distribution,
    plot_sentiment_distribution,
    plot_top_terms,
    plot_wordcloud,
)

__all__ = [
    # preprocessing
    "load_data", "load_sample_data", "clean_dataset", "clean_text",
    "preprocess_texts", "engineer_features", "split_dataset",
    "run_preprocessing_pipeline",
    # sentiment analysis
    "textblob_sentiment", "vader_sentiment", "apply_rule_based",
    "build_tfidf_vectorizer", "build_logistic_regression_pipeline",
    "build_svm_pipeline", "build_naive_bayes_pipeline",
    "build_random_forest_pipeline", "train_model", "compute_metrics",
    "evaluate_model", "save_model", "load_model", "run_training_pipeline",
    # visualization
    "plot_sentiment_distribution", "plot_rating_distribution",
    "plot_review_length_distribution", "plot_wordcloud", "plot_top_terms",
    "plot_confusion_matrix", "plot_model_comparison",
    "interactive_sentiment_over_time", "interactive_restaurant_comparison",
    "interactive_review_explorer",
]
