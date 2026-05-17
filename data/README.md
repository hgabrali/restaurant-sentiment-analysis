————–––├──└──├──├──├──└──└──–# Data Directory

This directory stores raw and processed restaurant review datasets.

---

## Data Sources

1. **Yelp Open Dataset** — https://www.yelp.com/dataset (restaurant subset)
2. **Google Maps Reviews** — Scraped using BeautifulSoup/requests (see `src/scraper.py`)
3. **TripAdvisor** — Public restaurant reviews dataset
4. **Synthetic Data** — Generated via `src/data_preprocessing.load_sample_data()`

---

## Data Dictionary

| Column | Type | Description | Example |
|---|---|---|---|
| review_text | string | Full customer review | 'Amazing food and service!' |
| rating | int (1-5) | Star rating given by reviewer | 4 |
| sentiment | string | Derived label | positive |
| restaurant_name | string | Name of the reviewed restaurant | La Bella Italia |
| cuisine_type | string | Cuisine category | Italian |
| reviewer_id | string | Anonymised reviewer ID | user_0042 |
| helpful_votes | int | Number of helpful votes | 12 |
| review_date | datetime | Date of submission | 2023-06-15 |

### Derived Features

| Column | Formula | Description |
|---|---|---|
| review_length | len(review_text) | Character count |
| word_count | len(review_text.split()) | Word count |
| avg_word_length | mean(len(w) for w in tokens) | Average word length |
| exclamation_count | count('!') | Enthusiasm proxy |
| question_count | count('?') | Complaint/inquiry proxy |
| uppercase_ratio | uppercase_chars / total_chars | Shouting / emphasis proxy |
| cleaned_text | preprocessed text | Input for ML models |

---

## Label Assignment

| Star Rating | Sentiment Label |
|---|---|
| 1 - 2 stars | negative |
| 3 stars | neutral |
| 4 - 5 stars | positive |

---

## Directory Structure

```
data/
├── raw/                    # Original unmodified data
|   └── reviews.csv
├── processed/              # Cleaned and engineered data
|   ├── train.csv
|   ├── val.csv
|   └── test.csv
└── README.md               # This file
```

---

## Statistics (Synthetic Dataset)

| Metric | Value |
|---|---|
| Total Reviews | 1,000 |
| Positive | 300 (30%) |
| Neutral | 150 (15%) |
| Negative | 250 (25%) |
| Avg Review Length | ~55 words |
| Unique Restaurants | 6 |
| Date Range | 2022-01 to 2022-07 |

---

> Raw data files are not committed to avoid size issues.
> Download from the sources above and place in `data/raw/`.
