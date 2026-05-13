# Bank Marketing Subscription Prediction

English | [中文](README_zh.md)

Predict whether a bank client will subscribe to a term deposit based on phone marketing campaign data, using LightGBM with 5-fold stratified cross-validation.

## Dataset

Based on the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), which contains data from a Portuguese bank's phone marketing campaigns.

| File | Description |
|------|-------------|
| `data/train.csv` | Training set with target label `subscribe` |
| `data/test.csv` | Test set without target label |
| `data/submission.csv` | Submission template |

### Feature Overview

**Client Information**: age, job, marital, education, default, housing, loan

**Campaign Contact**: contact, month, day_of_week, duration, campaign, pdays, previous, poutcome

**Economic Indicators**: emp_var_rate, cons_price_index, cons_conf_index, lending_rate3m, nr_employed

## Project Structure

```
bank-marketing-prediction/
├── src/
│   ├── main.py           # LightGBM pipeline (main model)
│   └── baseline.py       # Linear Regression baseline + visualization
├── data/                 # Dataset files
├── figures/              # Generated visualizations
├── output/               # Prediction results
├── requirements.txt
├── .gitignore
└── README.md
```

## Feature Engineering

| Feature | Method | Rationale |
|---------|--------|-----------|
| `contacted_before` | `pdays == 999 → 0, else 1` | Whether the client was previously contacted |
| `month_sin/cos` | Cyclic encoding | Capture monthly seasonality without false ordinality |
| `day_sin/cos` | Cyclic encoding | Same for day of week |
| `total_contacts` | `campaign + previous` | Total contact attempts across campaigns |
| `emp_cons_ratio` | `emp_var_rate / cons_conf_index` | Economic indicator interaction |
| `age_group` | Binning: young/mid/senior/elder | Non-linear age effect |
| `high_contact_freq` | `campaign > 5 → 1` | Flag over-contacted clients |
| `previous_success` | `poutcome == success → 1` | Historical success indicator |

## Models

### Main Model: LightGBM

- 5-fold stratified cross-validation with early stopping (patience=50)
- Class imbalance handled via `scale_pos_weight`
- Preprocessing pipeline: `SimpleImputer` + `StandardScaler` (numeric), `SimpleImputer` + `OneHotEncoder` (categorical)

### Baseline: Linear Regression

- Train/validation split (80/20)
- Generates diagnostic plots: feature importance, residual plot, PR curve, confusion matrix

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run LightGBM pipeline
python src/main.py

# Run baseline model + visualization
python src/baseline.py
```

## Results

The dataset is highly imbalanced (86.9% "no" vs 13.1% "yes"), making accuracy a misleading metric — a naive "predict all no" already achieves 86.9% accuracy. The real challenge is identifying the minority positive class.

### Model Comparison

| Metric | Linear Regression (Baseline) | LightGBM (5-Fold CV) |
|--------|------------------------------|----------------------|
| ROC-AUC | — | 0.797 |
| Accuracy | 86.44% | — |
| Precision | 45.98% | 41.79% |
| Recall | 13.42% | 58.74% |
| F1 Score | 20.78% | 48.84% |

### Key Insights

**ROC-AUC = 0.80** proves the model has genuine discriminative power — it can distinguish potential subscribers from non-subscribers far better than random.

**The default 0.5 threshold fails** on imbalanced data. Because `scale_pos_weight=6.62` dampens raw output probabilities, no sample exceeds 0.5 → LightGBM predicts all "no" out of the box.

**After threshold tuning** (grid-search best = 0.250 by max F1), the model catches **58.7% of actual subscribers** — a **4.4× improvement** over the baseline's 13.4%. The F1 score more than doubles (0.21 → 0.49).

This mirrors real-world marketing: precision of ~42% means 4 out of 10 targeted leads convert, while recall of ~59% means you reach most of the interested audience. A marketing team would value this trade-off over the baseline.

### Output Files

| File | Model | Description |
|------|-------|-------------|
| `output/submission_result.csv` | LightGBM | 5-fold CV + tuned threshold predictions |
| `output/submission_baseline.csv` | Linear Regression | Single split predictions |

Visualization figures are saved to `figures/`:

## Tech Stack

- **Python 3.8+**
- **LightGBM** - Gradient boosting model
- **scikit-learn** - Preprocessing, metrics, pipeline
- **Pandas / NumPy** - Data manipulation
- **Matplotlib / Seaborn** - Visualization
