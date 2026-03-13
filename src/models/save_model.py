# src/models/save_model.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from data.preprocess import load_data, build_preprocessor, cap_outliers

def save_pipeline():
    print("Loading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = cap_outliers(
        X_train.copy(), X_test.copy(),
        ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']
    )

    with open('src/models/best_params.json', 'r') as f:
        best_params = json.load(f)

    best_params.update({
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })

    print("Fitting preprocessor...")
    preprocessor, _ = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    print("Training model...")
    feature_names = X_train.columns.tolist()
    X_train_proc  = pd.DataFrame(
        preprocessor.transform(X_train), columns=feature_names
    )
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train_proc, y_train)

    joblib.dump(preprocessor, 'api/preprocessor.joblib')
    joblib.dump(model,        'api/model.joblib')

    caps = X_train[['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']].quantile(0.99)
    artifacts = {
        'feature_names' : feature_names,
        'cap_cols'      : ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'],
        'cap_values'    : caps.to_dict(),
        'threshold'     : 0.77
    }
    with open('api/artifacts.json', 'w') as f:
        json.dump(artifacts, f, indent=2)

    print("Saved:")
    print("  api/preprocessor.joblib")
    print("  api/model.joblib")
    print("  api/artifacts.json")

if __name__ == '__main__':
    save_pipeline()