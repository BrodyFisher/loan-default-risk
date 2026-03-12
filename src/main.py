# src/main.py
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# Add src to path so submodule imports work cleanly
sys.path.append(os.path.dirname(__file__))

from data.preprocess       import load_data, cap_outliers, build_preprocessor
from models.train          import train_model
from models.evaluate       import (print_full_metrics, tune_threshold,
                                   plot_evaluation, ks_statistic)
from models.shap_explain   import compute_shap, plot_shap, print_shap_summary

# ---------------------------------------------------------------
# CONFIG — change behaviour here without touching pipeline code
# ---------------------------------------------------------------
CONFIG = {
    'data_path'       : './GiveMeSomeCredit/cs-training.csv',
    'test_size'       : 0.2,
    'random_state'    : 42,
    'optuna_trials'   : 50,
    'cap_cols'        : ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio'],
    'params_path'     : 'src/models/best_params.json',
    'graphs_dir'      : 'src/visualization/graphs/',
    'run_optuna'      : False,   # ← set True to re-tune, False to load saved params
}

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def elapsed(start):
    s = int(time.time() - start)
    return f"{s//60}m {s%60}s"

# ---------------------------------------------------------------
# STEP 1 — DATA
# ---------------------------------------------------------------
def step_data():
    section("STEP 1 — Load & Split Data")
    start = time.time()

    X, y = load_data(CONFIG['data_path'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = CONFIG['test_size'],
        random_state = CONFIG['random_state'],
        stratify     = y
    )
    X_train, X_test = cap_outliers(
        X_train.copy(), X_test.copy(), CONFIG['cap_cols']
    )

    print(f"  Train : {X_train.shape}  |  default rate: {y_train.mean():.3f}")
    print(f"  Test  : {X_test.shape}   |  default rate: {y_test.mean():.3f}")
    print(f"  Done  : {elapsed(start)}")
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------
# STEP 2 — BASELINE MODELS
# ---------------------------------------------------------------
def step_baselines(X_train, y_train):
    section("STEP 2 — Baseline Model Training")
    start = time.time()

    lr = LogisticRegression(
        class_weight = 'balanced',
        max_iter     = 1000,
        random_state = CONFIG['random_state']
    )
    lr_pipeline = train_model(lr, "Logistic Regression", X_train, y_train)

    lgbm = lgb.LGBMClassifier(
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 31,
        scale_pos_weight  = (y_train == 0).sum() / (y_train == 1).sum(),
        random_state      = CONFIG['random_state'],
        n_jobs            = -1,
        verbose           = -1
    )
    lgbm_pipeline = train_model(lgbm, "LightGBM (Default)", X_train, y_train)

    print(f"\n  Done  : {elapsed(start)}")
    return lr_pipeline, lgbm_pipeline

# ---------------------------------------------------------------
# STEP 3 — HYPERPARAMETER TUNING
# ---------------------------------------------------------------
def step_tuning(X_train, y_train):
    section("STEP 3 — Hyperparameter Tuning")
    start = time.time()

    if CONFIG['run_optuna']:
        from models.evaluate import optuna_tune
        best_params = optuna_tune(X_train, y_train, CONFIG['optuna_trials'])
        with open(CONFIG['params_path'], 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  Saved best params → {CONFIG['params_path']}")
    else:
        print(f"  Loading saved params from {CONFIG['params_path']}")
        with open(CONFIG['params_path'], 'r') as f:
            best_params = json.load(f)

    best_params.update({
        'scale_pos_weight' : (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state'     : CONFIG['random_state'],
        'n_jobs'           : -1,
        'verbose'          : -1
    })

    tuned_pipeline = train_model(
        lgb.LGBMClassifier(**best_params),
        "LightGBM (Tuned)", X_train, y_train
    )

    print(f"\n  Done  : {elapsed(start)}")
    return tuned_pipeline, best_params

# ---------------------------------------------------------------
# STEP 4 — EVALUATION
# ---------------------------------------------------------------
def step_evaluation(X_test, y_test, lr_pipeline, lgbm_pipeline, tuned_pipeline):
    section("STEP 4 — Evaluation")
    start = time.time()

    results = {
        'Logistic Regression' : lr_pipeline.predict_proba(X_test)[:, 1],
        'LightGBM'            : lgbm_pipeline.predict_proba(X_test)[:, 1],
        'LightGBM (Tuned)'    : tuned_pipeline.predict_proba(X_test)[:, 1],
    }

    # Full metrics at default threshold
    for name, y_prob in results.items():
        print_full_metrics(name, y_test, y_prob, threshold=0.5)

    # Threshold tuning on best model
    threshold_results, best_threshold = tune_threshold(
        y_test, results['LightGBM (Tuned)'], 'LightGBM (Tuned)'
    )

    # Re-print tuned model at optimal threshold
    print_full_metrics(
        'LightGBM (Tuned) — Optimal Threshold',
        y_test, results['LightGBM (Tuned)'],
        threshold=best_threshold
    )

    # Save evaluation plots
    plot_evaluation(y_test, results, threshold_results,
                    save_path=f"{CONFIG['graphs_dir']}evaluation.png")

    print(f"\n  Done  : {elapsed(start)}")
    return results, best_threshold

# ---------------------------------------------------------------
# STEP 5 — SHAP INTERPRETABILITY
# ---------------------------------------------------------------
def step_shap(X_train, X_test, y_train, best_params):
    section("STEP 5 — SHAP Interpretability")
    start = time.time()

    feature_names = X_train.columns.tolist()

    # Preprocess outside pipeline so SHAP gets direct model access
    preprocessor, _ = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    X_train_proc = pd.DataFrame(
        preprocessor.transform(X_train), columns=feature_names
    )
    X_test_proc = pd.DataFrame(
        preprocessor.transform(X_test), columns=feature_names
    )

    # Train model directly (not in pipeline)
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train_proc, y_train)

    explainer, shap_values = compute_shap(model, X_test_proc)
    print_shap_summary(shap_values, feature_names)
    plot_shap(shap_values, X_test_proc,
          y_test,
          feature_names, CONFIG['graphs_dir'])

    print(f"\n  Done  : {elapsed(start)}")

# ---------------------------------------------------------------
# STEP 6 — FINAL SUMMARY
# ---------------------------------------------------------------
def step_summary(results, y_test, best_threshold, total_start):
    section("PIPELINE COMPLETE — Final Summary")

    for name, y_prob in results.items():
        print(f"  {name:<30} "
              f"AUC-ROC: {__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y_test, y_prob):.4f}  |  "
              f"AUC-PR: {__import__('sklearn.metrics', fromlist=['average_precision_score']).average_precision_score(y_test, y_prob):.4f}  |  "
              f"KS: {ks_statistic(y_test, y_prob):.4f}")

    print(f"\n  Optimal threshold (LightGBM Tuned) : {best_threshold:.2f}")
    print(f"  Graphs saved to                    : {CONFIG['graphs_dir']}")
    print(f"  Best params saved to               : {CONFIG['params_path']}")
    print(f"\n  Total pipeline time : {elapsed(total_start)}")

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == '__main__':
    total_start = time.time()

    os.makedirs(CONFIG['graphs_dir'], exist_ok=True)

    X_train, X_test, y_train, y_test          = step_data()
    lr_pipeline, lgbm_pipeline                = step_baselines(X_train, y_train)
    tuned_pipeline, best_params               = step_tuning(X_train, y_train)
    results, best_threshold                   = step_evaluation(X_test, y_test,
                                                                lr_pipeline,
                                                                lgbm_pipeline,
                                                                tuned_pipeline)
    step_shap(X_train, X_test, y_train, best_params)
    step_summary(results, y_test, best_threshold, total_start)