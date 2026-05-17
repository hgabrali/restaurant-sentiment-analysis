"""
visualization.py
================
Reusable plotting utilities for the Restaurant Sentiment Analysis project.

  All functions accept a matplotlib *ax* parameter so callers can embed plots
  in larger figure grids (e.g. inside the Colab notebook).
  """

  from __future__ import annotations

    from typing import Dict, List, Optional, Sequence

      import matplotlib.pyplot as plt
        import numpy as np
          import pandas as pd
            from sklearn.metrics import confusion_matrix

              # ---------------------------------------------------------------------------
              # Colour palette (consistent across all charts)
              # ---------------------------------------------------------------------------
              PALETTE: Dict[str, str] = {
                    "positive": "#2ecc71",
                    "neutral": "#f39c12",
                    "negative": "#e74c3c",
              }
              LABEL_ORDER: List[str] = ["negative", "neutral", "positive"]


              # ---------------------------------------------------------------------------
              # 1. Sentiment distribution bar / pie
              # ---------------------------------------------------------------------------

              def plot_sentiment_distribution(
                    df: pd.DataFrame,
                    label_col: str = "sentiment",
                    title: str = "Sentiment Distribution",
                    kind: str = "bar",
                ax: Optional[plt.Axes] = None,
                figsize: tuple = (7, 4),
              ) -> plt.Axes:
                  """Plot sentiment class counts as a bar chart or pie chart.

                  Parameters
                  ----------
                  df : pd.DataFrame
                      DataFrame that contains a *label_col* column.
                  label_col : str
                      Name of the sentiment column.
                  title : str
                      Chart title.
              kind : {'bar', 'pie'}
                      Chart type.
                  ax : plt.Axes, optional
                      Existing axes to draw on. A new figure is created if *None*.
                            figsize : tuple
                        Figure size (only used when *ax* is None).

                            Returns
                            -------
                            plt.Axes
                            """
                        counts = df[label_col].value_counts().reindex(LABEL_ORDER, fill_value=0)
                        colors = [PALETTE.get(lbl, "#95a5a6") for lbl in counts.index]

                              if ax is None:
                                _, ax = plt.subplots(figsize=figsize)

                                    if kind == "pie":
                                      ax.pie(
                                                    counts.values,
                                                    labels=counts.index,
                                                    colors=colors,
                                                    autopct="%1.1f%%",
                                                    startangle=140,
                                      )
                                      ax.set_title(title, fontsize=13, fontweight="bold")
                                          else:
                                            bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
                                            ax.set_title(title, fontsize=13, fontweight="bold")
                                            ax.set_xlabel("Sentiment", fontsize=11)
                                            ax.set_ylabel("Count", fontsize=11)
                                            ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
                                            ax.spines[["top", "right"]].set_visible(False)

                                                return ax


                                            # ---------------------------------------------------------------------------
                                            # 2. Confusion matrix
                                            # ---------------------------------------------------------------------------

                                            def plot_confusion_matrix(
                                              y_true: Sequence[str],
                                              y_pred: Sequence[str],
                                              labels: Optional[List[str]] = None,
                                                  title: str = "Confusion Matrix",
                                              ax: Optional[plt.Axes] = None,
                                              figsize: tuple = (6, 5),
                                                  cmap: str = "Blues",
                                            ) -> plt.Axes:
                                                """Plot a normalised confusion matrix as a heatmap.

                                                Parameters
                                                ----------
                                                y_true : array-like of str
                                                    Ground-truth labels.
                                                y_pred : array-like of str
                                                    Predicted labels.
                                                labels : list of str, optional
                                                    Label order. Defaults to LABEL_ORDER.
                                                title : str
                                                    Chart title.
                                                ax : plt.Axes, optional
                                                    Existing axes; new figure created if *None*.
                                                          figsize : tuple
                                                          cmap : str
                                                              Matplotlib colormap name.

                                                          Returns
                                                          -------
                                                          plt.Axes
                                                          """
                                                          if labels is None:
                                                                    labels = LABEL_ORDER

                                                            cm = confusion_matrix(y_true, y_pred, labels=labels)
                                                            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

                                                                if ax is None:
                                                                  _, ax = plt.subplots(figsize=figsize)

                                                                  im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
                                                                  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                                                                  tick_marks = np.arange(len(labels))
                                                                  ax.set_xticks(tick_marks)
                                                                  ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
                                                                  ax.set_yticks(tick_marks)
                                                                  ax.set_yticklabels(labels, fontsize=10)
                                                                  ax.set_xlabel("Predicted", fontsize=11)
                                                                  ax.set_ylabel("True", fontsize=11)
                                                                  ax.set_title(title, fontsize=13, fontweight="bold")

                                                                      thresh = 0.5
                                                                  for i in range(len(labels)):
                                                                    for j in range(len(labels)):
                                                                      ax.text(
                                                                                        j, i,
                                                                        f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                                                                                        ha="center", va="center", fontsize=9,
                                                                        color="white" if cm_norm[i, j] > thresh else "black",
                                                                          )

                                                                              return ax


                                                                          # ---------------------------------------------------------------------------
                                                                          # 3. Top TF-IDF features per class
                                                                            # ---------------------------------------------------------------------------

                                                                            def plot_top_features(
                                                                                  vectorizer,
                                                                                  classifier,
                                                                              labels: Optional[List[str]] = None,
                                                                                  top_n: int = 15,
                                                                                  title: str = "Top TF-IDF Features per Class",
                                                                              figsize: tuple = (14, 5),
                                                                            ) -> plt.Figure:
                                                                                """Bar charts of the highest-weighted TF-IDF features for each class.

                                                                            Works with sklearn models that expose ``coef_`` (LR, LinearSVC).
                                                                                For models without ``coef_``, a ``NotImplementedError`` is raised.

                                                                                Parameters
                                                                                ----------
                                                                                vectorizer : fitted TfidfVectorizer
                                                                                classifier : fitted classifier with ``coef_`` attribute
                                                                                labels : list of str, optional
                                                                                    Class names matching ``classifier.classes_`` order.
                                                                                top_n : int
                                                                                    Number of features to show per class.
                                                                                          title : str
                                                                                              Overall figure title.
                                                                                          figsize : tuple

                                                                                          Returns
                                                                                          -------
                                                                                          plt.Figure
                                                                                          """
                                                                                      if not hasattr(classifier, "coef_"):
                                                                                        raise NotImplementedError(
                                                                                          f"{type(classifier).__name__} does not expose 'coef_'. "
                                                                                                      "Use Logistic Regression or LinearSVC."
                                                                                        )

                                                                                        feature_names = np.array(vectorizer.get_feature_names_out())
                                                                                            if labels is None:
                                                                                              labels = list(getattr(classifier, "classes_", LABEL_ORDER))

                                                                                              n_classes = len(labels)
                                                                                              fig, axes = plt.subplots(1, n_classes, figsize=figsize, sharey=False)
                                                                                                  if n_classes == 1:
                                                                                                    axes = [axes]
                                                                                                    
                                                                                                    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
                                                                                                    
                                                                                                        coefs = classifier.coef_
                                                                                                    for idx, (label, ax) in enumerate(zip(labels, axes)):
                                                                                                      row = coefs[idx] if coefs.ndim == 2 else coefs[0]
                                                                                                        top_indices = np.argsort(row)[-top_n:][::-1]
                                                                                                        top_features = feature_names[top_indices]
                                                                                                        top_weights = row[top_indices]
                                                                                                        
                                                                                                        color = PALETTE.get(label, "#3498db")
                                                                                                        ax.barh(top_features[::-1], top_weights[::-1], color=color, edgecolor="white")
                                                                                                        ax.set_title(label.capitalize(), fontsize=12, color=color, fontweight="bold")
                                                                                                        ax.set_xlabel("Weight", fontsize=10)
                                                                                                        ax.spines[["top", "right"]].set_visible(False)
                                                                                                        
                                                                                                        plt.tight_layout()
                                                                                                            return fig
                                                                                                        
                                                                                                        
                                                                                                        # ---------------------------------------------------------------------------
                                                                                                        # 4. Model performance leaderboard
                                                                                                        # ---------------------------------------------------------------------------
                                                                                                        
                                                                                                        def plot_leaderboard(
                                                                                                              leaderboard: pd.DataFrame,
                                                                                                              metric: str = "f1_weighted",
                                                                                                              title: str = "Model Leaderboard",
                                                                                                          ax: Optional[plt.Axes] = None,
                                                                                                          figsize: tuple = (9, 5),
                                                                                                        ) -> plt.Axes:
                                                                                                            """Horizontal bar chart of model performance from the results DataFrame.
                                                                                                        
                                                                                                            Parameters
                                                                                                            ----------
                                                                                                            leaderboard : pd.DataFrame
                                                                                                                Output of ``run_training_pipeline`` with at least ``model`` and
                                                                                                                *metric* columns.
                                                                                                            metric : str
                                                                                                        Column to visualise (e.g. ``'f1_weighted'``, ``'f1_macro'``, ``'accuracy'``).
                                                                                                            title : str
                                                                                                            ax : plt.Axes, optional
                                                                                                            figsize : tuple
                                                                                                        
                                                                                                            Returns
                                                                                                            -------
                                                                                                            plt.Axes
                                                                                                            """
                                                                                                        df = leaderboard.sort_values(metric, ascending=True).copy()
                                                                                                        
                                                                                                            if ax is None:
                                                                                                              _, ax = plt.subplots(figsize=figsize)
                                                                                                              
                                                                                                              colors = ["#3498db" if v >= 0.90 else "#e67e22" if v >= 0.75 else "#e74c3c"
                                                                                                                for v in df[metric]]
                                                                                                                  
                                                                                                                  bars = ax.barh(df["model"], df[metric], color=colors, edgecolor="white")
                                                                                                                  ax.set_xlim(0, 1.05)
                                                                                                                  ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
                                                                                                                  ax.set_title(title, fontsize=13, fontweight="bold")
                                                                                                                  ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
                                                                                                                  ax.axvline(0.90, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
                                                                                                                  ax.spines[["top", "right"]].set_visible(False)
                                                                                                                  
                                                                                                                      return ax
                                                                                                                  
                                                                                                                  
                                                                                                                  # ---------------------------------------------------------------------------
                                                                                                                  # 5. Sentiment trend over time
                                                                                                                  # ---------------------------------------------------------------------------
                                                                                                                  
                                                                                                                  def plot_sentiment_trend(
                                                                                                                        df: pd.DataFrame,
                                                                                                                        date_col: str = "review_date",
                                                                                                                        label_col: str = "sentiment",
                                                                                                                        freq: str = "ME",
                                                                                                                    title: str = "Sentiment Trend Over Time",
                                                                                                                    ax: Optional[plt.Axes] = None,
                                                                                                                    figsize: tuple = (12, 5),
                                                                                                                  ) -> plt.Axes:
                                                                                                                      """Line chart of monthly sentiment proportions.
                                                                                                                  
                                                                                                                      Parameters
                                                                                                                      ----------
                                                                                                                      df : pd.DataFrame
                                                                                                                  Must contain *date_col* (datetime) and *label_col* (str).
                                                                                                                      date_col : str
                                                                                                                          Name of the date column.
                                                                                                                      label_col : str
                                                                                                                          Name of the sentiment column.
                                                                                                                      freq : str
                                                                                                                  Resampling frequency (default ``'ME'`` = month end).
                                                                                                                      title, ax, figsize : see other functions.
                                                                                                                  
                                                                                                                      Returns
                                                                                                                      -------
                                                                                                                      plt.Axes
                                                                                                                      """
                                                                                                                  tmp = df.copy()
                                                                                                                  tmp[date_col] = pd.to_datetime(tmp[date_col])
                                                                                                                  tmp = tmp.set_index(date_col)
                                                                                                                  
                                                                                                                  monthly = (
                                                                                                                    tmp.groupby([pd.Grouper(freq=freq), label_col])
                                                                                                                    .size()
                                                                                                                    .unstack(fill_value=0)
                                                                                                                  )
                                                                                                                  monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
                                                                                                                  
                                                                                                                      if ax is None:
                                                                                                                        _, ax = plt.subplots(figsize=figsize)
                                                                                                                        
                                                                                                                            for label in LABEL_ORDER:
                                                                                                                                      if label in monthly_pct.columns:
                                                                                                                                        ax.plot(
                                                                                                                                                          monthly_pct.index,
                                                                                                                                          monthly_pct[label],
                                                                                                                                                          marker="o",
                                                                                                                                                          markersize=4,
                                                                                                                                          label=label.capitalize(),
                                                                                                                                          color=PALETTE[label],
                                                                                                                                                          linewidth=1.8,
                                                                                                                                        )
                                                                                                                                        
                                                                                                                                        ax.set_title(title, fontsize=13, fontweight="bold")
                                                                                                                                        ax.set_xlabel("Month", fontsize=11)
                                                                                                                                        ax.set_ylabel("Percentage (%)", fontsize=11)
                                                                                                                                        ax.legend(title="Sentiment", fontsize=10)
                                                                                                                                        ax.spines[["top", "right"]].set_visible(False)
                                                                                                                                        plt.xticks(rotation=30, ha="right")
                                                                                                                                        
                                                                                                                                            return ax
