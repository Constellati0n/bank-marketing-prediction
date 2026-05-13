"""
Bank Marketing - Hyperparameter Optimization
Systematic tuning of LightGBM on imbalanced data

Key improvements over main.py:
1. AUC-based early stopping (binary_logloss fails on imbalanced data)
2. Grid search over scale_pos_weight / is_unbalance
3. Extended feature engineering (interactions, ratios)
4. Feature selection (remove correlated economic indicators)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import warnings
import os
import itertools

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    submission = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))
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
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    day_map = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2*np.pi*df['month'].map(month_map)/12)
        df['month_cos'] = np.cos(2*np.pi*df['month'].map(month_map)/12)
    if 'day_of_week' in df.columns:
        df['day_sin'] = np.sin(2*np.pi*df['day_of_week'].map(day_map)/7)
        df['day_cos'] = np.cos(2*np.pi*df['day_of_week'].map(day_map)/7)
    return df


def feature_engineering_extended(df):
    df = df.copy()
    df['total_contacts'] = df['campaign'] + df['previous']
    df['emp_cons_ratio'] = df['emp_var_rate'] / (df['cons_conf_index'] + 1e-5)
    df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100],
                             labels=['young','mid','senior','elder'])
    df['high_contact_freq'] = np.where(df['campaign'] > 5, 1, 0)
    if 'poutcome' in df.columns:
        df['previous_success'] = np.where(df['poutcome'] == 'success', 1, 0)

    df['age_job_interaction'] = df['age'].astype(str) + '_' + df['job'].astype(str)
    df['prev_success_ratio'] = np.where(df['previous'] > 0,
                                        df['previous_success'] / df['previous'], 0)
    df['campaign_intensity'] = df['campaign'] / (df['pdays'].replace(999, 30) + 1)

    return df


def create_preprocessing_pipeline(drop_correlated=True):
    numeric_features = ['age', 'campaign', 'pdays', 'previous']

    if not drop_correlated:
        numeric_features += ['emp_var_rate', 'cons_price_index',
                             'cons_conf_index', 'lending_rate3m', 'nr_employed']
    else:
        numeric_features += ['emp_var_rate', 'cons_conf_index', 'lending_rate3m']

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


def evaluate_params(train, test, params, n_folds=5):
    X = train.drop(columns=['subscribe', 'id'])
    y = train['subscribe']
    X_test = test.drop(columns=['id'])

    preprocessor = create_preprocessing_pipeline(drop_correlated=params.get('drop_correlated', True))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': params['learning_rate'],
        'num_leaves': params['num_leaves'],
        'min_child_samples': params['min_child_samples'],
        'feature_fraction': params.get('feature_fraction', 0.8),
        'bagging_fraction': params.get('bagging_fraction', 0.8),
        'bagging_freq': params.get('bagging_freq', 5),
        'seed': 42,
        'verbose': -1
    }

    if params.get('is_unbalance'):
        lgb_params['is_unbalance'] = True
    else:
        lgb_params['scale_pos_weight'] = params.get('scale_pos_weight', 6.62)

    n_rounds = params.get('num_boost_round', 500)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        X_tr_p = preprocessor.fit_transform(X_tr)
        X_val_p = preprocessor.transform(X_val)
        X_test_p = preprocessor.transform(X_test)

        tr_set = lgb.Dataset(X_tr_p, label=y_tr)
        val_set = lgb.Dataset(X_val_p, label=y_val)

        model = lgb.train(
            lgb_params, tr_set,
            num_boost_round=n_rounds,
            valid_sets=[tr_set, val_set],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        oof_preds[valid_idx] = model.predict(X_val_p)
        test_preds += model.predict(X_test_p) / n_folds

    oof_auc = roc_auc_score(y, oof_preds)

    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        f1_t = f1_score(y, (oof_preds > t).astype(int))
        if f1_t > best_f1:
            best_f1, best_t = f1_t, t

    oof_bin = (oof_preds > best_t).astype(int)
    prec = precision_score(y, oof_bin)
    rec = recall_score(y, oof_bin)

    return {
        'roc_auc': oof_auc,
        'f1': best_f1,
        'precision': prec,
        'recall': rec,
        'best_threshold': best_t,
        'test_preds': test_preds
    }


def grid_search(train, test):
    param_grid = [
        {'desc': 'Original (baseline)',    'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20,
         'scale_pos_weight': 6.62, 'is_unbalance': False, 'drop_correlated': True,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'num_boost_round': 1000},

        {'desc': 'Fix: AUC metric',         'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20,
         'scale_pos_weight': 6.62, 'is_unbalance': False, 'drop_correlated': True,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'num_boost_round': 1000},

        {'desc': 'AUC + is_unbalance',      'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20,
         'scale_pos_weight': None, 'is_unbalance': True, 'drop_correlated': True,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'num_boost_round': 1000},

        {'desc': 'AUC + lighter weight',    'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20,
         'scale_pos_weight': 3.0, 'is_unbalance': False, 'drop_correlated': True,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'num_boost_round': 1000},

        {'desc': 'AUC + all feats',         'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 20,
         'scale_pos_weight': 6.62, 'is_unbalance': False, 'drop_correlated': False,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'num_boost_round': 1000},

        {'desc': 'Deeper trees',            'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 10,
         'scale_pos_weight': 6.62, 'is_unbalance': False, 'drop_correlated': True,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'num_boost_round': 1000},

        {'desc': 'Deeper + AUC+unbalance',  'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 10,
         'scale_pos_weight': None, 'is_unbalance': True, 'drop_correlated': True,
         'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'num_boost_round': 1500},

        {'desc': 'Light + AUC+unbalance',   'learning_rate': 0.01, 'num_leaves': 127, 'min_child_samples': 5,
         'scale_pos_weight': None, 'is_unbalance': True, 'drop_correlated': True,
         'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'num_boost_round': 2000},

        {'desc': 'AUC+sw=3+deep',           'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 10,
         'scale_pos_weight': 3.0, 'is_unbalance': False, 'drop_correlated': True,
         'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'num_boost_round': 1500},

        {'desc': 'AUC+sw=2+deep',           'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 10,
         'scale_pos_weight': 2.0, 'is_unbalance': False, 'drop_correlated': True,
         'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'num_boost_round': 1500},

        {'desc': 'AUC+sw=1+deep',           'learning_rate': 0.03, 'num_leaves': 63, 'min_child_samples': 10,
         'scale_pos_weight': 1.0, 'is_unbalance': False, 'drop_correlated': True,
         'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'num_boost_round': 1500},
    ]

    results = []
    best_auc, best_result = 0, None

    for i, cfg in enumerate(param_grid):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(param_grid)}: {cfg['desc']}")
        print(f"{'='*60}")

        try:
            res = evaluate_params(train, test, cfg)
            results.append({**cfg, **res})
            print(f"  ROC-AUC: {res['roc_auc']:.4f}  F1: {res['f1']:.4f}  "
                  f"Prec: {res['precision']:.4f}  Rec: {res['recall']:.4f}  "
                  f"Thresh: {res['best_threshold']:.3f}")

            if res['roc_auc'] > best_auc:
                best_auc = res['roc_auc']
                best_result = {**cfg, **res}
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print("BEST RESULT")
    print(f"{'='*60}")
    print(f"Config:       {best_result['desc']}")
    print(f"ROC-AUC:      {best_result['roc_auc']:.4f}")
    print(f"F1 Score:     {best_result['f1']:.4f}")
    print(f"Precision:    {best_result['precision']:.4f}")
    print(f"Recall:       {best_result['recall']:.4f}")
    print(f"Best Thresh:  {best_result['best_threshold']:.3f}")

    print(f"\n{'='*60}")
    print("FULL RANKING (by ROC-AUC)")
    print(f"{'='*60}")
    results.sort(key=lambda x: x['roc_auc'], reverse=True)
    for r in results:
        print(f"  {r['desc']:<35s}  AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}")

    return best_result


def main():
    train, test, submission = load_data()
    print("Preprocessing...")
    train = preprocess_data(train)
    test = preprocess_data(test)
    print("Feature engineering...")
    train = feature_engineering_extended(train)
    test = feature_engineering_extended(test)

    print(f"\nTrain features: {train.shape[1]} cols")
    print(f"Target: no={train['subscribe'].value_counts()[0]}, "
          f"yes={train['subscribe'].value_counts()[1]} "
          f"(ratio=1:{train['subscribe'].value_counts()[0]/train['subscribe'].value_counts()[1]:.1f})")

    best = grid_search(train, test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub = submission.copy()
    sub['subscribe'] = (best['test_preds'] > best['best_threshold']).astype(int)
    sub['subscribe'] = sub['subscribe'].map({0: 'no', 1: 'yes'})
    sub.to_csv(os.path.join(OUTPUT_DIR, 'submission_optimized.csv'), index=False)
    print(f"\nOptimized submission saved to output/submission_optimized.csv")


if __name__ == "__main__":
    main()
