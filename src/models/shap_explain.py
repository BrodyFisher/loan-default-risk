# src/models/shap_explain.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

from data.preprocess import load_data, build_preprocessor, cap_outliers
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------
# 1. LOAD DATA & TRAIN TUNED MODEL
# ---------------------------------------------------------------
def load_and_train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = cap_outliers(
        X_train.copy(), X_test.copy(),
        ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']
    )

    # Load saved Optuna params — no need to re-tune
    with open('src/models/best_params.json', 'r') as f:
        best_params = json.load(f)

    best_params.update({
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    })

    # Fit preprocessor on train
    preprocessor, _ = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Transform both splits
    feature_names = X_train.columns.tolist()
    X_train_proc  = pd.DataFrame(preprocessor.transform(X_train), columns=feature_names)
    X_test_proc   = pd.DataFrame(preprocessor.transform(X_test),  columns=feature_names)

    # Train model on processed data directly (SHAP needs direct model access, not pipeline)
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train_proc, y_train)

    print("Model trained successfully")
    return model, X_train_proc, X_test_proc, y_train, y_test, feature_names

# ---------------------------------------------------------------
# 2. COMPUTE SHAP VALUES
# ---------------------------------------------------------------
def compute_shap(model, X_test_proc):
    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test_proc)
    print(f"SHAP values shape: {shap_values.shape}")
    return explainer, shap_values

# ---------------------------------------------------------------
# 3. PLOTS
# ---------------------------------------------------------------
def plot_shap(shap_values, X_test_proc, y_test, feature_names,
              save_dir='src/visualization/graphs/'):

    # --- 3a. Global: Beeswarm (best overall summary) ---
    print("\nPlotting beeswarm...")
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.title('SHAP Beeswarm — Global Feature Impact', fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(f'{save_dir}shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved shap_beeswarm.png")

    # --- 3b. Global: Bar (mean absolute SHAP — easiest to explain to stakeholders) ---
    print("Plotting bar summary...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title('SHAP Bar — Mean Absolute Feature Importance', fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(f'{save_dir}shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved shap_bar.png")

    # --- 3c. Dependence plots for top 3 features ---
    print("Plotting dependence plots...")
    # Mean absolute SHAP to find top features
    mean_shap   = np.abs(shap_values.values).mean(axis=0)
    top_features = np.argsort(mean_shap)[::-1][:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat_idx in enumerate(top_features):
        feat_name = feature_names[feat_idx]
        axes[i].scatter(
            X_test_proc.iloc[:, feat_idx],
            shap_values.values[:, feat_idx],
            alpha=0.3, s=8,
            c=shap_values.values[:, feat_idx],
            cmap='coolwarm'
        )
        axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[i].set_xlabel(feat_name)
        axes[i].set_ylabel('SHAP Value')
        axes[i].set_title(f'Dependence: {feat_name}')

    plt.suptitle('SHAP Dependence Plots — Top 3 Features', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{save_dir}shap_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved shap_dependence.png")

    # --- 3d. Local: Waterfall plots — one correct & one wrong prediction ---
    print("Plotting waterfall (local explanations)...")

    # Use positional indices, not dataframe index values
    y_test_arr = np.array(y_test.values)  # .values ensures positional alignment
    explainer_vals = shap_values.values

    tp_indices = np.where(y_test_arr == 1)[0]  # positional
    fp_indices = np.where(y_test_arr == 0)[0]  # positional

    # Pick highest confidence correct default prediction
    default_shap_sums = explainer_vals[tp_indices].sum(axis=1)
    tp_idx = tp_indices[np.argmax(default_shap_sums)]

    # Pick a false positive — good customer flagged as risky
    fp_shap_sums = explainer_vals[fp_indices].sum(axis=1)
    fp_idx = fp_indices[np.argmax(fp_shap_sums)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    plt.sca(axes[0])
    shap.plots.waterfall(shap_values[tp_idx], max_display=10, show=False)
    axes[0].set_title('True Positive — Actual Defaulter\n(correctly flagged)', fontsize=11)

    plt.sca(axes[1])
    shap.plots.waterfall(shap_values[fp_idx], max_display=10, show=False)
    axes[1].set_title('False Positive — Good Customer\n(wrongly flagged)', fontsize=11)

    plt.suptitle('SHAP Waterfall — Local Explanations', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved shap_waterfall.png")

# ---------------------------------------------------------------
# 4. PRINT GLOBAL SUMMARY TABLE
# ---------------------------------------------------------------
def print_shap_summary(shap_values, feature_names):
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    summary   = pd.DataFrame({
        'Feature'         : feature_names,
        'Mean |SHAP|'     : mean_shap,
        'Mean SHAP'       : shap_values.values.mean(axis=0),  # direction matters
    }).sort_values('Mean |SHAP|', ascending=False)

    print(f"\n{'='*55}")
    print("  Global SHAP Feature Importance")
    print(f"{'='*55}")
    print(summary.to_string(index=False, float_format='{:.4f}'.format))

# ---------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------
if __name__ == '__main__':
    import subprocess
    subprocess.run(['pip', 'install', 'shap', '-q'])

    model, X_train_proc, X_test_proc, y_train, y_test, feature_names = load_and_train()
    explainer, shap_values = compute_shap(model, X_test_proc)
    print_shap_summary(shap_values, feature_names)
    plot_shap(shap_values, X_test_proc, y_test, feature_names)

    print("\nDone. Check src/visualization/graphs/ for all SHAP plots.")