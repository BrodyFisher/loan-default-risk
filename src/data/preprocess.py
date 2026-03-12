# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

SENTINEL_COLS = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]

def load_data(data_path='./GiveMeSomeCredit/cs-training.csv'):
    df = pd.read_csv(data_path, index_col=0)

    # Fix sentinel values
    for col in SENTINEL_COLS:
        df[col] = df[col].replace([96, 98], np.nan)

    # Fix age = 0
    df['age'] = df['age'].replace(0, np.nan)

    X = df.drop(columns='SeriousDlqin2yrs')
    y = df['SeriousDlqin2yrs']
    return X, y

def cap_outliers(X_train, X_test, cap_cols):
    caps = X_train[cap_cols].quantile(0.99)
    for col in cap_cols:
        X_train[col] = X_train[col].clip(upper=caps[col])
        X_test[col]  = X_test[col].clip(upper=caps[col])
    return X_train, X_test

def build_preprocessor(X):
    numeric_features = X.columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
    return preprocessor, numeric_features