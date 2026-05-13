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

The dataset is highly imbalanced (86.9% "no" vs 13.1% "yes"), making accuracy a misleading metric — a naive "predict all no" baseline already achieves 86.9% accuracy. The real challenge is identifying the minority positive class.

### Model Comparison

| Metric | Linear Regression (Baseline) | LightGBM (5-Fold CV) |
|--------|------------------------------|----------------------|
| Accuracy | 86.44% | 86.88% |
| Precision | 45.98% | — |
| Recall | 13.42% | — |
| F1 Score | 20.78% | — |

> LightGBM applies `scale_pos_weight=6.62` to counter class imbalance. The per-fold accuracy is stable (86.87%–86.89%) with early stopping at ~5 rounds, indicating fast convergence without overfitting.

The baseline's poor recall (13.4%) shows that a simple linear model cannot effectively identify potential subscribers, highlighting the value of LightGBM's gradient boosting and feature engineering.

### Output Files

| File | Model | Description |
|------|-------|-------------|
| `output/submission_result.csv` | LightGBM | 5-fold CV averaged predictions |
| `output/submission_baseline.csv` | Linear Regression | Single split predictions |

Visualization figures are saved to `figures/`:

## Tech Stack

- **Python 3.8+**
- **LightGBM** - Gradient boosting model
- **scikit-learn** - Preprocessing, metrics, pipeline
- **Pandas / NumPy** - Data manipulation
- **Matplotlib / Seaborn** - Visualization
