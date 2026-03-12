# src/models/evaluate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------
# 1. FULL METRICS
# ---------------------------------------------------------------
def print_full_metrics(name, y_test, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n{'='*50}")
    print(f"  {name}  (threshold={threshold:.2f})")
    print(f"{'='*50}")
    print(f"  AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  AUC-PR   : {average_precision_score(y_test, y_prob):.4f}")
    print(f"  KS Stat  : {ks_statistic(y_test, y_prob):.4f}")
    print(f"\n  Confusion Matrix (threshold={threshold:.2f})")
    print(f"  {'':20} Predicted 0   Predicted 1")
    print(f"  {'Actual 0':20} {tn:>10,}   {fp:>10,}")
    print(f"  {'Actual 1':20} {fn:>10,}   {tp:>10,}")
    print(f"\n  Recall (Sensitivity) : {tp/(tp+fn):.4f}  ← catching actual defaults")
    print(f"  Precision            : {tp/(tp+fp):.4f}  ← of flagged, how many real")
    print(f"  Specificity          : {tn/(tn+fp):.4f}  ← correctly cleared non-defaults")
    print(f"  False Positive Rate  : {fp/(fp+tn):.4f}  ← good customers wrongly denied")

def ks_statistic(y_true, y_prob):
    # KS = max separation between TPR and FPR curves — industry standard in credit
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return (tpr - fpr).max()

# ---------------------------------------------------------------
# 2. THRESHOLD TUNING
# ---------------------------------------------------------------
def tune_threshold(y_test, y_prob, model_name):
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results.append({'threshold': t, 'recall': recall, 'precision': precision, 'f1': f1})

    results_df = pd.DataFrame(results)
    best_f1_row = results_df.loc[results_df['f1'].idxmax()]

    print(f"\n{'='*50}")
    print(f"  Threshold Tuning — {model_name}")
    print(f"{'='*50}")
    print(f"  Default threshold (0.50) F1 : {results_df[results_df['threshold'].between(0.49,0.51)]['f1'].values[0]:.4f}")
    print(f"  Best threshold             : {best_f1_row['threshold']:.2f}")
    print(f"  Best F1                    : {best_f1_row['f1']:.4f}")
    print(f"  Recall at best threshold   : {best_f1_row['recall']:.4f}")
    print(f"  Precision at best threshold: {best_f1_row['precision']:.4f}")

    return results_df, best_f1_row['threshold']

# ---------------------------------------------------------------
# 3. OPTUNA HYPERPARAMETER TUNING
# ---------------------------------------------------------------
def optuna_tune(X_train, y_train, n_trials=50):
    print(f"\n{'='*50}")
    print(f"  Optuna Tuning — {n_trials} trials")
    print(f"{'='*50}")

    def objective(trial):
        params = {
            'n_estimators'      : trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves'        : trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples' : trial.suggest_int('min_child_samples', 20, 100),
            'feature_fraction'  : trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction'  : trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq'      : trial.suggest_int('bagging_freq', 1, 7),
            'reg_alpha'         : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda'        : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight'  : (y_train == 0).sum() / (y_train == 1).sum(),
            'random_state'      : 42,
            'n_jobs'            : -1,
            'verbose'           : -1
        }

        model = lgb.LGBMClassifier(**params)
        cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring='average_precision', n_jobs=-1)
        return score.mean()

    sampler = TPESampler(seed=42)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best AUC-PR : {study.best_value:.4f}")
    print(f"  Best params : ")
    for k, v in study.best_params.items():
        print(f"    {k:<25} {v}")

    return study.best_params

# ---------------------------------------------------------------
# 4. PLOTS
# ---------------------------------------------------------------
def plot_evaluation(y_test, results, threshold_results, save_path='src/visualization/graphs/evaluation.png'):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    colors = {'Logistic Regression': 'steelblue', 'LightGBM': 'tomato', 'LightGBM (Tuned)': 'seagreen'}

    # ROC Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=colors[name])
    ax1.plot([0,1],[0,1],'k--', alpha=0.4)
    ax1.set_title('ROC Curve'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.legend(fontsize=8)

    # Precision-Recall Curves
    ax2 = fig.add_subplot(gs[0, 1])
    for name, y_prob in results.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, label=f"{name} ({ap:.3f})", color=colors[name])
    ax2.axhline(y_test.mean(), color='k', linestyle='--', alpha=0.4, label='Random')
    ax2.set_title('Precision-Recall Curve'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.legend(fontsize=8)

    # Threshold Tuning — LightGBM Tuned
    ax3 = fig.add_subplot(gs[0, 2])
    td  = threshold_results
    ax3.plot(td['threshold'], td['recall'],    label='Recall',    color='tomato')
    ax3.plot(td['threshold'], td['precision'], label='Precision', color='steelblue')
    ax3.plot(td['threshold'], td['f1'],        label='F1',        color='seagreen', linewidth=2)
    best_t = td.loc[td['f1'].idxmax(), 'threshold']
    ax3.axvline(best_t, color='black', linestyle='--', alpha=0.6, label=f'Best t={best_t:.2f}')
    ax3.set_title('Threshold Tuning (LightGBM Tuned)')
    ax3.set_xlabel('Threshold'); ax3.legend(fontsize=8)

    # Score distributions
    ax4 = fig.add_subplot(gs[1, :2])
    tuned_probs = results['LightGBM (Tuned)']
    ax4.hist(tuned_probs[y_test == 0], bins=50, alpha=0.6, color='steelblue', label='Non-Default', density=True)
    ax4.hist(tuned_probs[y_test == 1], bins=50, alpha=0.6, color='tomato',    label='Default',     density=True)
    ax4.set_title('Predicted Probability Distribution (LightGBM Tuned)')
    ax4.set_xlabel('Predicted Probability'); ax4.set_ylabel('Density')
    ax4.legend()

    # KS Table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    table_data = [[name,
                   f"{roc_auc_score(y_test, yp):.4f}",
                   f"{average_precision_score(y_test, yp):.4f}",
                   f"{ks_statistic(y_test, yp):.4f}"]
                  for name, yp in results.items()]
    table = ax5.table(
        cellText=table_data,
        colLabels=['Model', 'AUC-ROC', 'AUC-PR', 'KS'],
        loc='center', cellLoc='center'
    )
    table.scale(1, 2)
    ax5.set_title('Summary', pad=20)

    plt.suptitle('Model Evaluation Dashboard', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    from train import load_data, build_preprocessor, cap_outliers, train_model
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # --- Load ---
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = cap_outliers(
        X_train.copy(), X_test.copy(),
        ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']
    )

    # --- Train baseline models ---
    lr_pipeline = train_model(
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Logistic Regression", X_train, y_train
    )
    lgbm_pipeline = train_model(
        lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "LightGBM", X_train, y_train
    )

    # --- Optuna tuning ---
    best_params = optuna_tune(X_train, y_train, n_trials=50)
    best_params['scale_pos_weight'] = (y_train==0).sum()/(y_train==1).sum()
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1

    tuned_pipeline = train_model(
        lgb.LGBMClassifier(**best_params),
        "LightGBM (Tuned)", X_train, y_train
    )

    # --- Probabilities ---
    results = {
        'Logistic Regression' : lr_pipeline.predict_proba(X_test)[:, 1],
        'LightGBM'            : lgbm_pipeline.predict_proba(X_test)[:, 1],
        'LightGBM (Tuned)'    : tuned_pipeline.predict_proba(X_test)[:, 1],
    }

    # --- Full metrics at default threshold ---
    for name, y_prob in results.items():
        print_full_metrics(name, y_test, y_prob, threshold=0.5)

    # --- Threshold tuning on best model ---
    threshold_results, best_threshold = tune_threshold(
        y_test, results['LightGBM (Tuned)'], 'LightGBM (Tuned)'
    )

    # --- Re-print tuned model at best threshold ---
    print_full_metrics('LightGBM (Tuned)', y_test,
                       results['LightGBM (Tuned)'], threshold=best_threshold)

    # --- Plots ---
    plot_evaluation(y_test, results, threshold_results)