# Software Bug Prediction

A machine learning project that predicts software defects using CK object-oriented metrics and Halstead complexity metrics from 21 PROMISE and NASA MDP benchmark datasets.

The project includes a machine learning training pipeline, an ensemble prediction system, a FastAPI REST API, a Streamlit web interface, and tools for batch prediction and evaluation.

---

## Architecture

```text
Input Software Metrics
          |
   RobustScaler
          |
   Feature Selection
   (63 selected features)
          |
   +------+------+ 
   |      |      |
  XGBoost LightGBM RandomForest
   |      |      |
   +------+------+ 
          |
   Out-of-Fold Predictions
          |
 Logistic Regression
    (Meta-Learner)
          |
 Isotonic Calibration
          |
 Decision Threshold (0.430)
          |
   Buggy / Not Buggy
```

### Key Design Decisions

- **Ensemble learning:** Combines XGBoost, LightGBM, and Random Forest using a Logistic Regression meta-learner.
- **Feature selection:** Uses XGBoost feature importance to select 63 features from the original 64-feature input.
- **Data preprocessing:** Uses RobustScaler, fitted on the training split only.
- **Class imbalance handling:** Applies SMOTE to training data, including training folds during cross-validation, rather than validation folds.
- **Hyperparameter optimization:** Uses Optuna to tune the base models.
- **Probability calibration:** Uses isotonic regression fitted on the calibration split.
- **Threshold selection:** Selects the decision threshold using calibration data rather than final test labels.
- **Evaluation:** Reports final performance on a held-out test set.

---

## Final Model Performance

The final evaluation uses a held-out test set of **2,532 samples**. The decision threshold of **0.430** was selected using a separate calibration set.

| Metric | Value |
|---|---:|
| Overall Accuracy | 79.30% |
| Macro F1-Score | 0.7852 |
| Buggy F1-Score | 0.7441 |
| Not-Buggy F1-Score | 0.8263 |
| AUC-ROC | 0.8465 |
| Decision Threshold | 0.430 |

### Confusion Matrix

| Actual / Predicted | Not Buggy | Buggy |
|---|---:|---:|
| Not Buggy | 1,246 | 239 |
| Buggy | 285 | 762 |

The model's results reflect a trade-off between detecting defective modules and avoiding false alarms. Performance may vary across datasets and software projects.

*Metrics are taken from `training_log_final.txt` for the final reported evaluation.*

---

## Dataset Overview

The project uses 21 PROMISE repository and NASA MDP benchmark datasets, with **16,877 samples after deduplication**.

| Dataset Group | Datasets |
|---|---|
| Apache projects | ant-1.7, camel-1.6, log4j-1.1, lucene-2.0, poi-2.0, xalan-2.4, xerces-1.2, xerces-1.3 |
| Eclipse / IDE | jedit-3.2, jedit-4.2 |
| Other open-source projects | velocity-1.6, synapse-1.0, synapse-1.2, tomcat |
| Small projects | data_arc, data_ivy-2.0, data_prop-6, data_redaktor |
| NASA MDP | kc1, pc1, JM1 |

These benchmark datasets contain software metrics and defect labels used to train and evaluate defect prediction models.

---

## Project Structure

```text
softwarebugprediction/
├── train.py
├── preprocessing.py
├── inference.py
├── predict.py
├── app.py
├── api.py
├── streamlit_app.py
├── requirements.txt
├── dataset/
├── tests/
├── results/
├── thesis_assets/
├── generate_thesis_results.py
├── generate_thesis_tables.py
└── generate_thesis_assets.py
```

| File / Directory | Purpose |
|---|---|
| `train.py` | Model training, hyperparameter tuning, ensemble construction, calibration, and evaluation |
| `preprocessing.py` | Shared preprocessing and feature engineering |
| `inference.py` | Shared prediction logic and loading of saved model artifacts |
| `predict.py` | Command-line batch prediction |
| `app.py` | FastAPI REST API |
| `api.py` | Alternate API entry point |
| `streamlit_app.py` | Streamlit user interface |
| `dataset/` | Benchmark CSV datasets |
| `tests/` | Unit tests |
| `results/` | Generated evaluation and thesis outputs |
| `thesis_assets/` | Generated figures and visual assets |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/anshikasrivastavbst-a11y/softwarebugprediction.git
cd softwarebugprediction
```

### 2. Create a virtual environment (recommended)

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

---

## How to Run

### Train the model

```bash
python train.py
```

Training may take time because it includes hyperparameter optimization and ensemble training. Run this only when you intend to retrain the model.

### Run batch prediction

```bash
python predict.py "dataset/ant-1.7 (1).csv"
```

Use a valid CSV path supported by the prediction script.

### Start the FastAPI REST API

```bash
uvicorn app:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

### Start the Streamlit application

```bash
streamlit run streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v
```

### Generate thesis outputs

```bash
python generate_thesis_results.py
python generate_thesis_tables.py
python generate_thesis_assets.py
```

These scripts generate evaluation outputs, tables, and figures. Check their implementation before running them to confirm whether they use saved model artifacts or perform additional computation.

---

## Live Demo

**Software Bug Prediction System**

https://softwarebugprediction.onrender.com

The hosted demo may be unavailable if the hosting service is sleeping, restarting, or has encountered a deployment issue.

---

## Known Limitations

1. **Benchmark generalization:** Performance on PROMISE and NASA MDP datasets does not guarantee similar performance on newer projects, different programming languages, or real-world development environments.

2. **Dataset limitations:** Benchmark datasets may contain historical project-specific patterns. Deduplication and careful splitting help, but do not by themselves establish generalization to entirely unseen projects.

3. **Threshold dependence:** The decision threshold of `0.430` was selected using a separate calibration split. Classification metrics such as precision, recall, F1-score, and accuracy depend on the selected threshold. AUC-ROC evaluates ranking performance across thresholds.

4. **Ensemble validation:** The pipeline uses out-of-fold predictions to train the meta-learner. The relationship between fold-trained base models and the final base models retrained on the full training split should be considered when interpreting results.

5. **Class imbalance:** SMOTE can help address class imbalance, but synthetic samples may not fully represent the distribution of real defective modules.

6. **Research and portfolio use:** The project is intended for educational, research, and portfolio demonstration. It has not been established as a production-ready defect-triage system and requires external validation before operational use.

---

## Future Improvements

- Evaluate generalization using project-wise or time-based splits.
- Validate the ensemble and calibration methodology with additional experiments.
- Compare performance against simpler baseline models.
- Add monitoring for prediction quality and data drift.
- Improve input validation and API error handling.
- Evaluate the model on external datasets not used during development.

---

## Disclaimer

This project is an experimental software defect prediction system. Its predictions are estimates based on software metrics and historical benchmark labels; they should not replace code reviews, automated testing, static analysis, or engineering judgment.

---

## Author

**Anshika Srivastav**

GitHub: https://github.com/anshikasrivastavbst-a11y
