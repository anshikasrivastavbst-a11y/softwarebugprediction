# Software Bug Prediction

A machine learning project that predicts software defects using CK (Chidamber–Kemerer) object-oriented metrics and Halstead complexity metrics from 21 PROMISE and NASA MDP benchmark datasets.

The project includes a model training pipeline, a stacking ensemble, probability calibration, batch prediction, a FastAPI REST API, and a Streamlit web interface.

## Live Demo

**Streamlit Web Application:**  
https://softwarebugprediction.onrender.com

The hosted application may take some time to start if the hosting service is idle.

## System Architecture

```text
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

## Key Features

- **Ensemble Learning:** Combines XGBoost, LightGBM, and Random Forest with Logistic Regression as the meta-learner.
- **Feature Engineering:** Creates additional software complexity and coupling metrics.
- **Feature Selection:** Uses 63 selected features from a 64-feature scaled input.
- **Data Preprocessing:** Processes numerical features, handles missing values, and removes selected noisy features.
- **Class Imbalance Handling:** Uses SMOTE during training.
- **Hyperparameter Optimization:** Uses Optuna to tune model parameters.
- **Probability Calibration:** Uses isotonic regression to calibrate prediction probabilities.
- **Threshold-Based Classification:** Applies a decision threshold of 0.430.
- **Multiple Interfaces:** Supports a Streamlit web interface, a FastAPI REST API, and batch prediction.

## Model Performance

The reported evaluation uses a held-out test set of 2,532 samples. The decision threshold was selected using a separate calibration set.

| Metric | Value |
|---|---:|
| Accuracy | 79.30% |
| Macro F1-Score | 0.7852 |
| Buggy F1-Score | 0.7441 |
| Not-Buggy F1-Score | 0.8263 |
| ROC-AUC | 0.8465 |
| Decision Threshold | 0.430 |

### Confusion Matrix

| Actual / Predicted | Not Buggy | Buggy |
|---|---:|---:|
| Not Buggy | 1,246 | 239 |
| Buggy | 285 | 762 |

These results represent the reported benchmark evaluation. Performance may vary across datasets and software projects.

## Dataset Overview

The project uses 21 PROMISE and NASA MDP benchmark datasets, with 16,877 samples reported after deduplication.

| Dataset Group | Datasets |
|---|---|
| Apache Projects | ant-1.7, camel-1.6, log4j-1.1, lucene-2.0, poi-2.0, xalan-2.4, xerces-1.2, xerces-1.3 |
| Eclipse / IDE | jedit-3.2, jedit-4.2 |
| Other Open-Source Projects | velocity-1.6, synapse-1.0, synapse-1.2, tomcat |
| Additional Projects | data_arc, data_ivy-2.0, data_prop-6, data_redaktor |
| NASA MDP | kc1, pc1, JM1 |

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, Random Forest
- **Data Processing:** Pandas, NumPy
- **Model Optimization:** Optuna
- **API:** FastAPI
- **Web Interface:** Streamlit
- **Model Persistence:** Joblib
- **Testing:** Pytest

## Project Structure

```text
softwarebugprediction/
├── .streamlit/
├── dataset/
├── tests/
├── api.py
├── app.py
├── streamlit_app.py
├── train.py
├── preprocessing.py
├── inference.py
├── predict.py
├── experiment.py
├── requirements.txt
├── MODEL_CARD.md
└── README.md
```

The repository may also contain saved model artifacts, test files, logs, and prediction outputs. Add them to this tree if they are present and relevant to the project.

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

### Run the Streamlit Web Application

```powershell
py -m streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal.

### Run the FastAPI REST API

Start the API:

```powershell
py -m uvicorn app:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

The FastAPI application has been tested locally with feature input. A separate public FastAPI deployment is not currently claimed.

An alternate API entry point is available through `api.py`:

```powershell
py -m uvicorn api:app --reload
```

Use the entry point that matches the API implementation you intend to run.

### Run Batch Prediction

```powershell
py predict.py "dataset/ant-1.7 (1).csv"
```

Replace the example path with a CSV file supported by the prediction script.

### Run Tests

```powershell
py -m pytest tests/ -v
```

### Retrain the Model

```powershell
py train.py
```

Training may take considerable time because it includes model optimization and ensemble training. Run this command only when you intend to retrain the model.

## API Usage

The FastAPI application provides the following endpoints:

- `GET /` — returns API information.
- `POST /predict` — returns a defect prediction, status, probability, and applied decision threshold.

Use the interactive Swagger documentation at `http://127.0.0.1:8000/docs` while the local API server is running.

The API expects software metric inputs compatible with the project's preprocessing pipeline. Supply meaningful feature values for useful predictions. Missing values may be filled during preprocessing, so incomplete inputs can produce less reliable results.

## Known Limitations

- **Generalization:** Benchmark results do not guarantee similar performance on unseen software projects.
- **Dataset Limitations:** Historical datasets may contain project-specific patterns.
- **Threshold Dependence:** Classification outcomes depend on the selected decision threshold.
- **Class Imbalance:** SMOTE-generated samples may not fully represent real-world defects.
- **Input Validation:** API input validation and error handling can be improved.
- **Production Readiness:** The model requires further external validation before use in production defect-triage workflows.

## Future Improvements

- Evaluate generalization using project-wise or time-based splits.
- Compare performance against simpler baseline models.
- Improve API input validation and error handling.
- Monitor prediction quality and data drift.
- Evaluate the model on external datasets.

## Disclaimer

This project is intended for educational, research, and portfolio purposes. Predictions are estimates based on software metrics and historical benchmark labels. They should not replace code reviews, automated testing, static analysis, or engineering judgment.

## Author

**Anshika Srivastav**

GitHub: https://github.com/anshikasrivastavbst-a11y
