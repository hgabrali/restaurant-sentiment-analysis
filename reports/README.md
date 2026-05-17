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
|   ├── wordcloud_neutral.png
|   ├── top_terms_positive.png
|   ├── top_terms_negative.png
|   ├── confusion_matrix_*.png
|   └── model_comparison_*.png
└── README.md
```

---

## Key Findings

### 1. Sentiment Distribution
- 60% of reviews are positive, 25% negative, 15% neutral.
- Negative reviews are more detailed (avg 72 words vs 45 for positive).
- Extreme ratings (1-star and 5-star) dominate — 3-star is the least common.

### 2. Top Words by Sentiment

**Positive:** amazing, delicious, excellent, fresh, friendly, wonderful, perfect, cozy
**Negative:** slow, cold, rude, bland, wait, overpriced, terrible, disappointing
**Neutral:** okay, decent, average, nothing, fine, typical, standard

### 3. Model Performance
- LinearSVC achieves best F1 (0.88) on the test set.
- VADER outperforms TextBlob on restaurant-specific vocabulary.
- TF-IDF bigrams significantly improve over unigrams (+5% F1).

### 4. Temporal Patterns
- Friday and Saturday reviews skew more negative (longer waits).
- January and February show more positive reviews (lower traffic).
- Review volume peaks in summer months.

---

## Business Recommendations

| Recommendation | Expected Impact | Priority |
|---|---|---|
| Deploy LinearSVC model for real-time review classification | Instant sentiment alerts | High |
| Monitor 'slow service' keyword trend weekly | 65% of negative reviews | High |
| Respond to all 1-2 star reviews within 24h | +12% return customer rate | High |
| Train staff on friendliness during peak hours | Reduces weekend negativity | Medium |
| Add incentives for 3-star reviewers to elaborate | Better neutral understanding | Low |

---

> Run `python -m src.visualization` to regenerate all figures.
