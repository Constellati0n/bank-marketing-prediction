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
│   ├── main.py           # LightGBM pipeline (optimized)
│   ├── baseline.py       # Linear Regression baseline + visualization
│   └── optimize.py       # Hyperparameter grid search (11 configs)
├── notebooks/
│   └── eda.ipynb         # Exploratory Data Analysis
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

- **Optimized parameters** (via `src/optimize.py` grid search over 11 configs)
- 5-fold stratified cross-validation + AUC-based early stopping (patience=30)
- `scale_pos_weight=1.0` — allows natural probability spread; imbalance handled by threshold tuning
- Deeper trees: `num_leaves=63`, `learning_rate=0.03`, `min_child_samples=10`
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

# Run hyperparameter search (11 configs)
python src/optimize.py
```

## Results

The dataset is highly imbalanced (86.9% "no" vs 13.1% "yes"), making accuracy a misleading metric — a naive "predict all no" already achieves 86.9% accuracy. The real challenge is identifying the minority positive class.

### Model Performance

| Metric | Linear Regression (Baseline) | LightGBM (Optimized) |
|--------|------------------------------|:---------------------:|
| ROC-AUC | — | **0.791** |
| Precision | 45.98% | **44.24%** |
| Recall | 13.42% | **56.10%** |
| F1 Score | 20.78% | **49.47%** |

### Output Files

| File | Model | Description |
|------|-------|-------------|
| `output/submission_result.csv` | LightGBM (Optimized) | 5-fold CV + tuned threshold |
| `output/submission_baseline.csv` | Linear Regression | Single split predictions |
| `output/submission_optimized.csv` | LightGBM (Best trial) | From `src/optimize.py` grid search |

Visualization figures are saved to `figures/`:

## Tech Stack

- **Python 3.8+**
- **LightGBM** - Gradient boosting model
- **scikit-learn** - Preprocessing, metrics, pipeline
- **Pandas / NumPy** - Data manipulation
- **Matplotlib / Seaborn** - Visualization
