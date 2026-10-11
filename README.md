# Software Bug Prediction

A machine learning project that predicts software defects using CK (Chidamber–Kemerer) object-oriented metrics and Halstead complexity metrics from 21 PROMISE and NASA MDP benchmark datasets.

The project includes a model training pipeline, a stacking ensemble, probability calibration, batch prediction, a FastAPI REST API, and a Streamlit web interface.

## Live Demo

**Streamlit Web Application:**  
https://softwarebugprediction.onrender.com

The hosted application may take time to start if the hosting service is idle.

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
  +----+----+
  |    |    |
 XGBoost   LightGBM   Random Forest
  |    |    |
  +----+----+
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

- **Ensemble learning:** Combines XGBoost, LightGBM, and Random Forest with Logistic Regression as the meta-learner.
- **Feature engineering:** Creates additional software complexity and coupling metrics.
- **Feature selection:** Uses 63 selected features from the original 64-feature model input.
- **Data preprocessing:** Handles missing values, removes selected noisy features, and scales numerical features.
- **Class imbalance handling:** Uses SMOTE in the training pipeline.
- **Hyperparameter optimization:** Uses Optuna to tune model parameters.
- **Probability calibration:** Uses isotonic regression to calibrate prediction probabilities.
- **Threshold-based classification:** Uses a decision threshold of 0.430.
- **Multiple interfaces:** Supports a Streamlit web interface, a FastAPI REST API, and batch prediction.

## Model Performance

The reported evaluation uses a held-out test set of 2,532 samples. The threshold was selected using a separate calibration set.

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
| Apache projects | ant-1.7, camel-1.6, log4j-1.1, lucene-2.0, poi-2.0, xalan-2.4, xerces-1.2, xerces-1.3 |
| Eclipse / IDE | jedit-3.2, jedit-4.2 |
| Other open-source projects | velocity-1.6, synapse-1.0, synapse-1.2, tomcat |
| Additional projects | data_arc, data_ivy-2.0, data_prop-6, data_redaktor |
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
├── model.pkl
├── scaler.pkl
├── selected_features.pkl
├── threshold.pkl
├── imputer.pkl
├── test.csv
├── predictions_output.csv
├── training_log_final.txt
├── MODEL_CARD.md
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/anshikasrivastavbst-a11y/softwarebugprediction.git
cd softwarebugprediction
```

### 2. Create a virtual environment

**Windows:**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Run the Streamlit application

```bash
py -m streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal.

### Run the FastAPI REST API

Start the API:

```bash
py -m uvicorn app:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

The API documentation is available locally while the server is running. The FastAPI API has been tested locally with feature input; a separate public API deployment is not currently claimed.

If you want to run the alternate API entry point, use:

```bash
py -m uvicorn api:app --reload
```

### Run batch prediction

```bash
py predict.py "dataset/ant-1.7 (1).csv"
```

Use a valid CSV file supported by the prediction script.

### Run tests

```bash
pytest tests/ -v
```

### Retrain the model

```bash
py train.py
```

Training can take considerable time because it includes model optimization and ensemble training. Run it only when you intend to retrain the model.

## API Usage

The FastAPI application provides a prediction endpoint:

- `POST /predict` — returns a defect prediction, status, probability, and decision threshold.
- `GET /` — returns API information.

Use the interactive Swagger documentation at `http://127.0.0.1:8000/docs` to inspect the available endpoints and test requests.

The API expects software metric inputs compatible with the project's preprocessing pipeline. Missing metrics may be filled with zeros, so meaningful feature values should be supplied for useful predictions.

## Known Limitations

- **Generalization:** Benchmark results do not guarantee similar performance on unseen software projects.
- **Dataset limitations:** Historical datasets may contain project-specific patterns.
- **Threshold dependence:** Classification results depend on the selected decision threshold.
- **Class imbalance:** SMOTE-generated samples may not fully represent real-world defects.
- **Input validation:** Input validation and error handling can be improved.
- **Production readiness:** The model has not been established as a production-ready defect-triage system and requires further external validation.

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
