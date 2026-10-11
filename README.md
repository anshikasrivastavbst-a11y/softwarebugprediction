# Software Bug Prediction

A machine learning project that predicts software defects from CK (Chidamber–Kemerer) object-oriented metrics and Halstead complexity metrics, trained on 21 PROMISE and NASA MDP benchmark datasets.

The project includes a full training pipeline, a stacking ensemble with probability calibration, batch prediction, a **FastAPI REST API** and a **Streamlit web interface**. Both the API and the web app are deployed publicly on Render.

## Table of Contents

- [Live Demo](#live-demo)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Dataset Overview](#dataset-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [API Usage](#api-usage)
- [Deployment](#deployment)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Author](#author)

## Live Demo

| Service | Link |
| ------- | ---- |
| Streamlit Web App | https://softwarebugprediction.onrender.com |
| FastAPI REST API (Swagger docs) | https://softwarebugprediction-api.onrender.com/docs |

Both services are hosted on Render's free tier. After a period of inactivity the service goes to sleep, so the **first request may take 30–60 seconds** to respond.

## System Architecture

```
Software Metrics
       |
Preprocessing & Feature Engineering
       |
RobustScaler
       |
Feature Selection (63 Features)
       |
  +----+----------+
  |    |          |
XGBoost  LightGBM  Random Forest
  |    |          |
  +----+----------+
       |
Logistic Regression
   Meta-Learner
       |
Isotonic Calibration
       |
Decision Threshold: 0.430
       |
Buggy / Not Buggy
```

Serving layer:

```
Client  ->  FastAPI (api.py)  ->  Preprocessing + Model (inference.py)  ->  JSON response
Browser ->  Streamlit (streamlit_app.py)  ->  Preprocessing + Model       ->  Prediction UI
```

## Key Features

- **Ensemble Learning:** XGBoost, LightGBM and Random Forest combined with a Logistic Regression meta-learner.
- **Feature Engineering:** Additional software complexity and coupling metrics derived from the raw inputs.
- **Feature Selection:** 63 selected features from a 64-feature scaled input.
- **Data Preprocessing:** Numerical feature handling, missing value imputation and removal of noisy features.
- **Class Imbalance Handling:** SMOTE applied during training.
- **Hyperparameter Optimization:** Optuna used to tune model parameters.
- **Probability Calibration:** Isotonic regression for calibrated probabilities.
- **Threshold-Based Classification:** Decision threshold of 0.430, selected on a separate calibration set.
- **Multiple Interfaces:** Streamlit web app, FastAPI REST API and batch prediction from CSV.
- **Public Deployment:** API and web app both hosted on Render.

## Model Performance

The evaluation uses a held-out test set of 2,532 samples. The decision threshold was selected on a separate calibration set, so the test set was not used for tuning.

| Metric             | Value  |
| ------------------ | ------ |
| Accuracy           | 79.30% |
| Macro F1-Score     | 0.7852 |
| Buggy F1-Score     | 0.7441 |
| Not-Buggy F1-Score | 0.8263 |
| ROC-AUC            | 0.8465 |
| Decision Threshold | 0.430  |

### Confusion Matrix

| Actual / Predicted | Not Buggy | Buggy |
| ------------------ | --------- | ----- |
| Not Buggy          | 1,246     | 239   |
| Buggy              | 285       | 762   |

These results represent the reported benchmark evaluation. Performance may vary across datasets and software projects.

## Dataset Overview

The project uses 21 PROMISE and NASA MDP benchmark datasets, with 16,877 samples reported after deduplication.

| Dataset Group              | Datasets                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------- |
| Apache Projects            | ant-1.7, camel-1.6, log4j-1.1, lucene-2.0, poi-2.0, xalan-2.4, xerces-1.2, xerces-1.3 |
| Eclipse / IDE              | jedit-3.2, jedit-4.2                                                                  |
| Other Open-Source Projects | velocity-1.6, synapse-1.0, synapse-1.2, tomcat                                        |
| Additional Projects        | data_arc, data_ivy-2.0, data_prop-6, data_redaktor                                    |
| NASA MDP                   | kc1, pc1, JM1                                                                         |

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Random Forest
- **Data Processing:** Pandas, NumPy
- **Model Optimization:** Optuna
- **API:** FastAPI, Uvicorn
- **Web Interface:** Streamlit
- **Model Persistence:** Joblib
- **Testing:** Pytest
- **Deployment:** Render

## Project Structure

```
softwarebugprediction/
├── .streamlit/                  # Streamlit configuration
├── dataset/                     # PROMISE / NASA MDP datasets
├── tests/                       # Pytest test suite
├── api.py                       # FastAPI application (deployed entry point)
├── app.py                       # Alternate API entry point
├── streamlit_app.py             # Streamlit web interface
├── train.py                     # Model training pipeline
├── preprocessing.py             # Preprocessing and feature engineering
├── inference.py                 # Shared inference logic
├── predict.py                   # Batch prediction script
├── experiment.py                # Experiments
├── generate_thesis_*.py         # Scripts that generate thesis tables, results and assets
├── model.pkl                    # Trained stacking model
├── scaler.pkl                   # Fitted RobustScaler
├── imputer.pkl                  # Fitted imputer
├── selected_features.pkl        # Selected feature names
├── threshold.pkl                # Decision threshold
├── requirements.txt
├── MODEL_CARD.md
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anshikasrivastavbst-a11y/softwarebugprediction.git
cd softwarebugprediction
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Streamlit Web Application

```bash
py -m streamlit run streamlit_app.py
```

Open the local URL shown in the terminal.

### FastAPI REST API

```bash
py -m uvicorn api:app --reload
```

Then open the interactive Swagger documentation at <http://127.0.0.1:8000/docs>.

### Batch Prediction

```bash
py predict.py "dataset/ant-1.7 (1).csv"
```

Replace the path with any CSV file supported by the prediction script.

### Retrain the Model

```bash
py train.py
```

Training can take a long time because it includes hyperparameter optimization and ensemble training. Run it only when you intend to retrain.

## API Usage

**Base URL (live):** `https://softwarebugprediction-api.onrender.com`

**Interactive docs:** `https://softwarebugprediction-api.onrender.com/docs`

### Endpoints

| Method | Endpoint   | Description                                                    |
| ------ | ---------- | -------------------------------------------------------------- |
| GET    | `/`        | Returns basic API information                                  |
| POST   | `/predict` | Returns the defect prediction, status, probability and threshold |

### POST /predict

Send a JSON object of software metrics. The keys must match the feature names expected by the preprocessing pipeline (see `selected_features.pkl` and the column names in `test.csv`).

**Example request (cURL):**

```bash
curl -X POST "https://softwarebugprediction-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
        "wmc": 12,
        "dit": 2,
        "noc": 0,
        "cbo": 8,
        "rfc": 25,
        "lcom": 10,
        "loc": 150
      }'
```

**Example request (Python):**

```python
import requests

url = "https://softwarebugprediction-api.onrender.com/predict"
payload = {
    "wmc": 12,
    "dit": 2,
    "noc": 0,
    "cbo": 8,
    "rfc": 25,
    "lcom": 10,
    "loc": 150,
}

response = requests.post(url, json=payload, timeout=90)
print(response.json())
```

The example above sends only 7 metrics (`wmc`, `dit`, `noc`, `cbo`, `rfc`, `lcom`, `loc`). Any feature that is not supplied is filled in by the imputer, so for reliable predictions send as many of the model's features as possible.

**Example response:**

```json
{
  "prediction": 0,
  "status": "Not Buggy",
  "probability": 0.2288,
  "applied_threshold": 0.43
}
```

| Field               | Meaning                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `prediction`        | `1` = Buggy, `0` = Not Buggy                                       |
| `status`            | Human-readable label                                               |
| `probability`       | Calibrated probability that the code is buggy (0 to 1)             |
| `applied_threshold` | Cut-off used to convert the probability into a class (0.43)        |

A prediction of `Buggy` is returned when `probability >= applied_threshold`.

> **Note:** Missing features are filled by the preprocessing imputer, so incomplete inputs can produce less reliable results. Always supply meaningful values for as many features as possible.

## Deployment

Both services are deployed on Render from this repository.

**FastAPI service (Web Service)**

| Setting         | Value                                         |
| --------------- | --------------------------------------------- |
| Language        | Python 3                                      |
| Branch          | `main`                                        |
| Build Command   | `pip install -r requirements.txt`             |
| Start Command   | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Instance Type   | Free                                          |

**Streamlit service (Web Service)** is deployed separately from the same repository.

Free-tier services sleep after a period of inactivity, so the first request after idle time is slow.

## Testing

```bash
py -m pytest tests/ -v
```

## Known Limitations

- **Generalization:** Benchmark results do not guarantee similar performance on unseen software projects.
- **Dataset Limitations:** Historical datasets may contain project-specific patterns.
- **Threshold Dependence:** Classification outcomes depend on the selected decision threshold.
- **Class Imbalance:** SMOTE-generated samples may not fully represent real-world defects.
- **Input Validation:** The `/predict` endpoint accepts a free-form JSON object. Unrecognised feature names are silently ignored and missing features are filled by the imputer, so malformed input can still return a `200` response with a prediction. A typed request schema is planned (see Future Improvements).
- **Hosting:** The free-tier deployment has cold starts and limited resources.
- **Production Readiness:** The model needs further external validation before use in real defect-triage workflows.

## Future Improvements

- Add a typed Pydantic request schema so Swagger shows every feature and invalid input is rejected with clear errors.
- Add a `GET /health` endpoint that reports whether the model has loaded.
- Version the API (for example `/v1/predict`).
- Evaluate generalization using project-wise or time-based splits.
- Compare performance against simpler baseline models.
- Monitor prediction quality and data drift.
- Evaluate the model on external datasets.

## Disclaimer

This project is intended for educational, research and portfolio purposes. Predictions are estimates based on software metrics and historical benchmark labels. They should not replace code reviews, automated testing, static analysis or engineering judgment.

## Author

**Anshika Srivastav**

GitHub: <https://github.com/anshikasrivastavbst-a11y>
