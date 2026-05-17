# Reports Directory

Generated analysis reports, figures, and model evaluation outputs.

---

## Directory Structure

```
reports/
├── figures/
|   ├── sentiment_distribution.png
|   ├── rating_distribution.png
|   ├── review_length_distribution.png
|   ├── wordcloud_positive.png
|   ├── wordcloud_negative.png
|   ├── top_terms_positive.png
|   ├── top_terms_negative.png
|   ├── top_terms_neutral.png
|   ├── confusion_matrix_*.png
|   └── model_comparison_*.png
└── README.md
```

---

## Key Findings

> All figures below are computed from the actual dataset (530 cleaned reviews, seed=42).
> > See the [Colab notebook](https://colab.research.google.com/drive/1DXn51z4XP4cVBx6h-3D797BfDDTUzaTk) for full reproducible analysis.
> >
> > ### 1. Sentiment Distribution
> >
> > - **57.4%** positive (304) | **26.8%** negative (142) | **15.8%** neutral (84)
> > - - `neutral` is the minority class — use **macro-F1** as the headline metric on real data.
> >   - - Extreme ratings (1-star and 5-star) dominate the distribution.
> >    
> >     - ### 2. Text Length Analysis
> >    
> >     - - Mean word count: negative ≈ 24 | neutral ≈ 26 | positive ≈ 24
> >       - - **Neutral** reviews are marginally longer on average — mixed-sentiment reviews carry more qualifying detail.
> >         - - Word-count vs rating correlation: **r = −0.170** (weak negative relationship).
> >          
> >           - ### 3. Top Words by Sentiment (TF-IDF, unigrams)
> >          
> >           - **Positive:** `every`, `experience`, `great`, `friend`, `wait`
> >           - **Negative:** `experience`, `wrong`, `almost`, `returning`, `certainly`
> >           - **Neutral:** (see notebook §10 for full chart)
> > 
### 4. Model Performance (Synthetic Dataset)

> ⚠️ Near-perfect ML scores are an **artifact of the synthetic phrase-pool data**, not a real result. On genuine restaurant reviews expect roughly **0.75–0.90 weighted F1**.
>
> | Model | Accuracy | F1 (weighted) |
> |---|---|---|
> | Logistic Regression | 1.0000 | 1.0000 |
> | LinearSVC | 1.0000 | 1.0000 |
> | Naive Bayes | 1.0000 | 1.0000 |
> | Random Forest | 1.0000 | 1.0000 |
> | XGBoost | 0.9906 | 0.9905 |
> | Rule: TextBlob | 0.9151 | 0.9092 |
> | Rule: VADER | 0.8396 | 0.7858 |
>
> - **TextBlob** outperforms VADER on this synthetic corpus (F1 0.91 vs 0.79).
> - - **XGBoost** is the only ML model below perfect, reflecting sensitivity to class imbalance and token sparsity.
>   - - TF-IDF bigrams are used by all ML pipelines (`ngram_range=(1,2)`, `max_features=10 000`).
>    
>     - ### 5. Temporal Patterns
>    
>     - - Monthly sentiment trend visible in interactive Plotly chart (notebook §12).
>       - - Day-of-week breakdown generated via ordered Categorical — chronologically correct.
>         - - Synthetic data is uniformly distributed across days; real data may show weekend skew.
>          
>           - ### 6. Restaurant & Cuisine Comparison
>          
>           - - **Best cuisine by avg rating:** Italian (3.57 ★)
>             - - **Worst cuisine by avg rating:** Indian (3.40 ★)
>               - - Restaurants: La Bella Italia, The Golden Spoon, Sakura Garden, El Rancho, The Rustic Table, Spice Route.
>                
>                 - ---
>
> ## Business Recommendations
>
> | Recommendation | Expected Impact | Priority |
> |---|---|---|
> | Deploy best classifier for real-time review classification | Instant sentiment alerts | High |
> | Use **macro-F1** as headline metric — not weighted F1 | Avoids misleading neutral-class scores | High |
> | Upload a real labelled CSV before evaluating model quality | Validates results beyond synthetic data | High |
> | Wrap LinearSVC with `CalibratedClassifierCV` | Enables confidence thresholding | Medium |
> | Respond to all 1–2 star reviews within 24 h | Improves return customer rate | Medium |
> | Run `GridSearchCV` on `val_df` | Optimises C, max_features, ngram_range | Medium |
> | Benchmark DistilBERT fine-tune on real data | Handles negation, sarcasm, emoji | Low |
>
> ---
>
> > Run `python -m src.visualization` to regenerate all figures.
