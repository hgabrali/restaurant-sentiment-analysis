"""
Restaurant Sentiment Analysis - Source Package
================================================
Public API for preprocessing and training pipelines.

Usage
-----
from src import run_preprocessing_pipeline, run_training_pipeline
from src import load_model, clean_text
"""

from .data_preprocessing import run_preprocessing_pipeline, clean_text
from .sentiment_analysis import run_training_pipeline, load_model
from .visualization import plot_sentiment_distribution, plot_confusion_matrix, plot_top_features

__all__ = [
      "run_preprocessing_pipeline",
      "run_training_pipeline",
      "clean_text",
      "load_model",
      "plot_sentiment_distribution",
      "plot_confusion_matrix",
      "plot_top_features",
]

__version__ = "2.0.0"
__author__ = "Hande Gabrali-Knobloch"
