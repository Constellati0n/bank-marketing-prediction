"""
Bank Marketing Subscription Prediction - Baseline Model
Linear Regression + Visualization (Confusion Matrix, PR Curve, Feature Importance)

The baseline uses the same raw features as the main LightGBM model but no
custom feature engineering, to provide a fair lower-bound comparison.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def load_and_preprocess_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    submission_ids = test['id'] if 'id' in test.columns else test.index

    train['subscribe'] = train['subscribe'].map({'yes': 1, 'no': 0})

    numeric_features = [
        'age', 'campaign', 'pdays', 'previous',
        'emp_var_rate', 'cons_price_index', 'cons_conf_index',
        'lending_rate3m', 'nr_employed'
    ]

    categorical_features = ['job', 'marital', 'education', 'default',
                            'housing', 'loan', 'contact', 'poutcome',
                            'month', 'day_of_week']

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    for df in [train, test]:
        if 'pdays' in df.columns:
            df.loc[df['pdays'] == 999, 'pdays'] = np.nan
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].replace('unknown', np.nan)
                df[col] = df[col].replace('other', np.nan)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X_train = train.drop('subscribe', axis=1)
    y_train = train['subscribe']
    X_test = test.copy()

    preprocessor.fit(X_train)
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)

    return X_train_prep, y_train, X_test_prep, submission_ids, all_feature_names


def train_and_evaluate(X_train, y_train):
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train_split, y_train_split)

    y_pred = model.predict(X_val)
    y_pred_class = (y_pred >= 0.5).astype(int)

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred_class)

    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation R2: {r2:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

    return model, X_val, y_val, y_pred, y_pred_class


def visualize_model(model, feature_names, X_val, y_val, y_pred, y_pred_class):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    coefs = model.coef_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Absolute_Value': np.abs(coefs)
    })

    top_features = feature_importance.sort_values('Absolute_Value', ascending=False).head(20)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='Coefficient', y='Feature',
                data=top_features.sort_values('Coefficient', ascending=False))
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importances.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    residuals = y_val - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.savefig(os.path.join(FIGURES_DIR, 'residual_plot.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, bins=30, kde=True)
    plt.axvline(0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.title('Prediction Distribution')
    plt.xlabel('Predicted Value')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'prediction_distribution.png'), dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    average_precision = average_precision_score(y_val, y_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, lw=2, color='navy',
             label=f'PR Curve (AP={average_precision:.2f})')
    plt.fill_between(recall, precision, alpha=0.2, color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'precision_recall_curve.png'), dpi=150)
    plt.close()

    cm = confusion_matrix(y_val, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\nClassification Metrics:")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")


def generate_submission(model, X_test, ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_pred_proba = model.predict(X_test)
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    y_pred_labels = np.where(y_pred_class == 1, 'yes', 'no')

    submission = pd.DataFrame({
        'id': ids,
        'subscribe': y_pred_labels
    })

    output_path = os.path.join(OUTPUT_DIR, 'submission_baseline.csv')
    submission.to_csv(output_path, index=False)
    print(f"Baseline submission saved: {output_path}")
    return submission


def main():
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, submission_ids, feature_names = load_and_preprocess_data()
    print(f"Preprocessed train shape: {X_train.shape}")
    print(f"Preprocessed test shape: {X_test.shape}")

    print("\nTraining baseline model...")
    model, X_val, y_val, y_pred, y_pred_class = train_and_evaluate(X_train, y_train)

    print("\nGenerating visualizations...")
    visualize_model(model, feature_names, X_val, y_val, y_pred, y_pred_class)

    print("\nGenerating submission...")
    submission = generate_submission(model, X_test, submission_ids)

    yes_pct = (submission['subscribe'] == 'yes').mean() * 100
    print(f"\nPredicted 'yes' ratio: {yes_pct:.2f}%")
    print("\n===== Baseline Complete =====")


if __name__ == "__main__":
    main()
