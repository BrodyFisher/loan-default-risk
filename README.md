# Loan Default Risk — End-to-End ML Pipeline

End-to-end machine learning pipeline for predicting loan defaults within two years, with a live FastAPI serving layer, Prometheus/Grafana monitoring, and SHAP interpretability.

---

## Stack

- **Modeling** — Scikit-learn, LightGBM, Optuna
- **Interpretability** — SHAP
- **API** — FastAPI, Uvicorn
- **Monitoring** — Prometheus, Grafana, Evidently AI
- **Infrastructure** — Docker, Docker Compose

---

## Results

| Model | AUC-ROC | AUC-PR | KS Stat |
|---|---|---|---|
| Logistic Regression | 0.859 | 0.385 | 0.559 |
| LightGBM (Default) | 0.861 | 0.380 | 0.577 |
| LightGBM (Tuned) | 0.869 | 0.401 | 0.584 |

---

## Project Structure

```
.
├── api/                        # FastAPI app and model artifacts
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Training, evaluation, SHAP, model saving
│   ├── visualization/          # EDA and evaluation plots
│   └── main.py                 # Pipeline entry point
├── monitoring/
│   ├── prometheus/             # Scrape config
│   ├── grafana/                # Dashboard and provisioning
│   └── evidently/reports/      # Generated drift reports
├── docker-compose.yml
└── Dockerfile
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/BrodyFisher/loan-default-risk.git
cd loan-default-risk
pip install -r requirements.txt
```

### 2. Set up Kaggle credentials

Create a `.env` file in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Get your API key from https://www.kaggle.com/settings → API → Create New Token.

### 3. Download the dataset

```python
from dotenv import load_dotenv
import os, kaggle

load_dotenv()
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('brycecf/give-me-some-credit-dataset', path='./GiveMeSomeCredit', unzip=True)
```

### 4. Run the ML pipeline

```bash
python src/main.py
```

### 5. Save the trained model

```bash
python src/models/save_model.py
```

### 6. Start the monitoring stack

```bash
docker compose up --build
```

---

## Services

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana login: `admin` / `admin`

---

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.5,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.3,
    "MonthlyIncome": 6000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
```

Each prediction returns a default probability, risk flag, and the top three SHAP-derived risk factors driving the decision.
