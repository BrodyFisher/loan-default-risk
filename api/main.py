# api/main.py
import json
import time
import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Loan Default Risk API", version="1.0.0")

# ---------------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------------
preprocessor = joblib.load('preprocessor.joblib')
model        = joblib.load('model.joblib')

with open('artifacts.json', 'r') as f:
    artifacts = json.load(f)

feature_names = artifacts['feature_names']
cap_cols      = artifacts['cap_cols']
cap_values    = artifacts['cap_values']
threshold     = artifacts['threshold']

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Store recent predictions for drift monitoring
prediction_log = []

# ---------------------------------------------------------------
# PROMETHEUS METRICS
# ---------------------------------------------------------------
REQUEST_COUNT = Counter(
    'loan_api_requests_total',
    'Total prediction requests',
    ['status']
)
REQUEST_LATENCY = Histogram(
    'loan_api_latency_seconds',
    'Prediction request latency',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_SCORE = Histogram(
    'loan_prediction_score',
    'Distribution of predicted default probabilities',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
DEFAULT_FLAGS = Counter(
    'loan_default_flags_total',
    'Total predictions flagged as default risk'
)
DRIFT_SCORE = Gauge(
    'loan_feature_drift_score',
    'Latest Evidently drift score',
    ['feature']
)

# ---------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------------
class BorrowerFeatures(BaseModel):
    RevolvingUtilizationOfUnsecuredLines : float = Field(..., ge=0, description="Credit utilization ratio")
    age                                  : int   = Field(..., ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse : int   = Field(..., ge=0)
    DebtRatio                            : float = Field(..., ge=0)
    MonthlyIncome                        : float = Field(None)
    NumberOfOpenCreditLinesAndLoans      : int   = Field(..., ge=0)
    NumberOfTimes90DaysLate              : int   = Field(..., ge=0)
    NumberRealEstateLoansOrLines         : int   = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse : int   = Field(..., ge=0)
    NumberOfDependents                   : float = Field(None)

class PredictionResponse(BaseModel):
    default_probability : float
    risk_flag           : bool
    risk_label          : str
    top_risk_factors    : list
    threshold_used      : float

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def preprocess_input(data: dict) -> pd.DataFrame:
    rename = {
        'NumberOfTime30_59DaysPastDueNotWorse': 'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60_89DaysPastDueNotWorse': 'NumberOfTime60-89DaysPastDueNotWorse',
    }
    df = pd.DataFrame([data]).rename(columns=rename)
    df = df[feature_names]

    for col in cap_cols:
        df[col] = df[col].clip(upper=cap_values[col])

    return df

def get_shap_explanation(df_proc: pd.DataFrame) -> list:
    sv        = explainer(df_proc)
    shap_vals = sv.values[0]
    top_idx   = np.argsort(np.abs(shap_vals))[::-1][:3]
    return [
        {
            'feature'    : feature_names[i],
            'shap_value' : round(float(shap_vals[i]), 4),
            'direction'  : 'increases risk' if shap_vals[i] > 0 else 'decreases risk'
        }
        for i in top_idx
    ]

# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM (Tuned)", "threshold": threshold}

@app.post("/predict", response_model=PredictionResponse)
def predict(borrower: BorrowerFeatures):
    start = time.time()
    try:
        raw     = borrower.model_dump()
        df      = preprocess_input(raw)
        df_proc = pd.DataFrame(
            preprocessor.transform(df), columns=feature_names
        )

        prob       = float(model.predict_proba(df_proc)[0][1])
        risk_flag  = prob >= threshold
        risk_label = "HIGH RISK" if risk_flag else "LOW RISK"

        top_factors = get_shap_explanation(df_proc)

        # Log for drift monitoring
        prediction_log.append({**raw, 'score': prob})
        if len(prediction_log) > 10000:
            prediction_log.pop(0)

        # Prometheus
        PREDICTION_SCORE.observe(prob)
        if risk_flag:
            DEFAULT_FLAGS.inc()
        REQUEST_COUNT.labels(status='success').inc()
        REQUEST_LATENCY.observe(time.time() - start)

        return PredictionResponse(
            default_probability = round(prob, 4),
            risk_flag           = risk_flag,
            risk_label          = risk_label,
            top_risk_factors    = top_factors,
            threshold_used      = threshold
        )

    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/drift")
def drift_report():
    if len(prediction_log) < 100:
        return {
            "message": f"Need at least 100 predictions for drift report, have {len(prediction_log)}"
        }

    ref_df  = pd.read_csv('GiveMeSomeCredit/cs-training.csv', index_col=0).sample(1000)
    curr_df = pd.DataFrame(prediction_log[-500:])

    shared_cols = [c for c in feature_names if c in curr_df.columns]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df[shared_cols], current_data=curr_df[shared_cols])

    # Update Prometheus drift gauges
    result = report.as_dict()
    for metric in result['metrics']:
        if 'drift_by_columns' in metric.get('result', {}):
            for feat, vals in metric['result']['drift_by_columns'].items():
                DRIFT_SCORE.labels(feature=feat).set(vals.get('statistic', 0))

    report.save_html('monitoring/evidently/reports/drift_report.html')
    return {
        "message": "Drift report saved",
        "path"   : "monitoring/evidently/reports/drift_report.html"
    }
