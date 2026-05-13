"""
Bank Marketing Subscription Prediction - Main Pipeline
LightGBM + 5-Fold Stratified Cross-Validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    submission = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print("Target distribution:")
    print(train['subscribe'].value_counts(normalize=True))

    return train, test, submission


def preprocess_data(df):
    df = df.copy()

    df['contacted_before'] = np.where(df['pdays'] == 999, 0, 1)

    if 'subscribe' in df.columns:
        df['subscribe'] = df['subscribe'].map({'no': 0, 'yes': 1})

    categorical_cols = ['job', 'marital', 'education', 'default',
                        'housing', 'loan', 'contact', 'poutcome']
    for col in categorical_cols:
        df[col] = df[col].replace('unknown', np.nan)
        df[col] = df[col].replace('other', np.nan)

    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'].map(month_map) / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'].map(month_map) / 12)

    if 'day_of_week' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'].map(day_map) / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'].map(day_map) / 7)

    return df


def feature_engineering(df):
    df = df.copy()

    df['total_contacts'] = df['campaign'] + df['previous']

    df['emp_cons_ratio'] = df['emp_var_rate'] / (df['cons_conf_index'] + 1e-5)

    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 30, 45, 60, 100],
                             labels=['young', 'mid', 'senior', 'elder'])

    df['high_contact_freq'] = np.where(df['campaign'] > 5, 1, 0)

    if 'poutcome' in df.columns:
        df['previous_success'] = np.where(df['poutcome'] == 'success', 1, 0)

    return df


def create_preprocessing_pipeline():
    numeric_features = ['age', 'campaign', 'pdays', 'previous',
                        'emp_var_rate', 'cons_price_index',
                        'cons_conf_index', 'lending_rate3m', 'nr_employed']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['job', 'marital', 'education', 'default',
                            'housing', 'loan', 'contact', 'poutcome', 'age_group']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor


def train_and_predict(train, test, submission):
    X = train.drop(columns=['subscribe', 'id'])
    y = train['subscribe']
    X_test = test.drop(columns=['id'])

    pos_weight = np.sum(y == 0) / np.sum(y == 1)
    print(f"Class ratio: 1:{pos_weight:.2f}")
    print(f"(Using scale_pos_weight=1.0 — threshold tuning handles imbalance instead)")

    preprocessor = create_preprocessing_pipeline()

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'min_child_samples': 10,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'scale_pos_weight': 1.0,
        'seed': 42,
        'verbose': -1
    }

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"\n======= Fold {fold + 1} =======")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        X_train_processed = preprocessor.fit_transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)
        X_test_processed = preprocessor.transform(X_test)

        train_set = lgb.Dataset(X_train_processed, label=y_train)
        valid_set = lgb.Dataset(X_valid_processed, label=y_valid)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=1500,
            valid_sets=[train_set, valid_set],
            callbacks=[lgb.early_stopping(30)]
        )

        oof_preds[valid_idx] = model.predict(X_valid_processed)
        test_preds += model.predict(X_test_processed) / n_folds

        valid_preds = (oof_preds[valid_idx] > 0.5).astype(int)
        fold_acc = accuracy_score(y_valid, valid_preds)
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")

    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    print(f"\nOverall OOF Accuracy: {oof_acc:.4f}")

    oof_binary = (oof_preds > 0.5).astype(int)
    precision = precision_score(y, oof_binary)
    recall = recall_score(y, oof_binary)
    f1 = f1_score(y, oof_binary)
    roc_auc = roc_auc_score(y, oof_preds)
    cm = confusion_matrix(y, oof_binary)
    tn, fp, fn, tp = cm.ravel()

    print("\n========== LightGBM - 5-Fold CV Evaluation ==========")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"Accuracy      : {oof_acc:.4f}  (naive all-no: 86.88%)")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Lift over naive: Recall shows we catch {recall*100:.1f}% of actual subscribers")
    print("========================================================")

    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = 0
    for t in thresholds:
        f1_t = f1_score(y, (oof_preds > t).astype(int))
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    oof_tuned = (oof_preds > best_thresh).astype(int)
    t_prec = precision_score(y, oof_tuned)
    t_rec = recall_score(y, oof_tuned)
    t_f1 = f1_score(y, oof_tuned)
    t_cm = confusion_matrix(y, oof_tuned)

    print(f"\n--- After Threshold Tuning (best={best_thresh:.3f}, max F1) ---")
    print(f"Precision     : {t_prec:.4f}")
    print(f"Recall        : {t_rec:.4f}")
    print(f"F1 Score      : {t_f1:.4f}")
    print(f"Confusion Matrix: {t_cm.ravel()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission['subscribe'] = (test_preds > best_thresh).astype(int)
    submission['subscribe'] = submission['subscribe'].map({0: 'no', 1: 'yes'})
    output_path = os.path.join(OUTPUT_DIR, 'submission_result.csv')
    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}")

    return submission


def main():
    train, test, submission = load_data()

    print("\nPreprocessing...")
    train = preprocess_data(train)
    test = preprocess_data(test)

    print("\nFeature engineering...")
    train = feature_engineering(train)
    test = feature_engineering(test)

    print("\nTraining model...")
    train_and_predict(train, test, submission)

    print("\n===== Pipeline Complete =====")


if __name__ == "__main__":
    main()
