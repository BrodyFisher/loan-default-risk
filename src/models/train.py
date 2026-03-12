# src/models/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Import from preprocess — single source of truth
from data.preprocess import load_data, build_preprocessor, cap_outliers

# --- Train & evaluate a model with cross-validation ---
def train_model(model, model_name, X_train, y_train):
    preprocessor, _ = build_preprocessor(X_train)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring={
            'roc_auc':           'roc_auc',
            'average_precision': 'average_precision'
        },
        return_train_score=False,
        n_jobs=-1
    )

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  AUC-ROC:  {cv_results['test_roc_auc'].mean():.4f}  ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"  AUC-PR:   {cv_results['test_average_precision'].mean():.4f}  ± {cv_results['test_average_precision'].std():.4f}")

    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == '__main__':
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = cap_outliers(
        X_train.copy(), X_test.copy(),
        ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']
    )

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_pipeline = train_model(lr, "Logistic Regression (Baseline)", X_train, y_train)

    lgbm = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_pipeline = train_model(lgbm, "LightGBM", X_train, y_train)