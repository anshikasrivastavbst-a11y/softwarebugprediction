# Software Bug Prediction

A machine learning pipeline for predicting software defects using CK object-oriented
metrics and Halstead complexity metrics from 21 PROMISE and NASA MDP benchmark datasets.

Developed as part of a Master's thesis project.

---

## Architecture

```
Input Features (63 CK / Halstead metrics)
        |
   RobustScaler
        |
  +-----+------+----------+
  |            |          |
XGBoost   LightGBM  RandomForest
  |            |          |
  +-----+------+----------+
        | (stacking OOF)
  LogisticRegression (meta-learner)
        |
  Isotonic Calibration
        |
  Threshold 0.430
        |
  Prediction (Buggy / Not Buggy)
```

**Key design decisions:**
- SMOTE balancing applied on training data only (never on validation or test folds)
- RobustScaler fitted on training set only (no leakage)
- Feature selection via XGBoost importance (dropped 1 zero-importance feature)
- Optuna hyperparameter tuning with 3-fold stratified CV per model
- Isotonic probability calibration for reliable bug probability scores

---

## Final Metrics

Evaluated on an untouched held-out test set of 2,532 samples. The decision threshold was selected using a separate calibration set, not the final test set.

| Metric | Value |
|---|---:|
| Overall Accuracy | 79.30% |
| Macro F1 | 0.7852 |
| Buggy F1 | 0.7441 |
| Buggy Recall | 72.78% |
| Not-Buggy F1 | 0.8263 |
| Not-Buggy Recall | 83.91% |
| AUC-ROC | 0.8465 |
| Decision Threshold | 0.430 |

The reported metrics correspond to the final evaluation recorded in `training_log_final.txt`. An earlier development run reported different values; those results should not be treated as the final evaluation of the currently saved model artifacts.

---

## Datasets

21 PROMISE repository and NASA MDP datasets, 16,877 samples after deduplication.

| Dataset Group    | Datasets |
|------------------|----------|
| Apache projects  | ant-1.7, camel-1.6, log4j-1.1, lucene-2.0, poi-2.0, xalan-2.4, xerces-1.2, xerces-1.3 |
| Eclipse/IDE      | jedit-3.2, jedit-4.2 |
| Other open-source| velocity-1.6, synapse-1.0, synapse-1.2, tomcat |
| Small projects   | data_arc, data_ivy-2.0, data_prop-6, data_redaktor |
| NASA MDP         | kc1, pc1, JM1 |

---

## Project Structure

```
softwarebugprediction/
|-- train.py                  # Full training pipeline
|-- preprocessing.py          # Shared preprocessing / feature engineering
|-- inference.py              # Shared prediction logic (used by all consumers)
|-- predict.py                # CLI batch prediction
|-- app.py                    # FastAPI REST API
|-- api.py                    # Alternate API entry point
|-- streamlit_app.py          # Streamlit web UI
|-- requirements.txt
|-- dataset/                  # 21 CSV dataset files
|-- tests/                    # 112 unit tests
|-- results/                  # Thesis evaluation outputs (generated)
|-- thesis_assets/            # Publication-quality figures (generated)
|-- generate_thesis_results.py
|-- generate_thesis_tables.py
|-- generate_thesis_assets.py
```

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train the model:**
```bash
python train.py
```

**Batch prediction from CSV:**
```bash
python predict.py dataset/ant-1.7\ \(1\).csv
```

**Start the REST API:**
```bash
uvicorn app:app --reload
# Visit http://127.0.0.1:8000/docs for interactive Swagger UI
```

**API usage example:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"wmc": 25, "dit": 3, "cbo": 12, "rfc": 45, "loc": 300}'
```

**Streamlit web UI:**
```bash
streamlit run streamlit_app.py
```

**Run tests:**
```bash
pytest tests/ -v
```

**Generate thesis outputs (frozen model, no retraining):**
```bash
python generate_thesis_results.py
python generate_thesis_tables.py
python generate_thesis_assets.py
```

---

## Known Limitations

1. **Benchmark generalisation:** Performance is measured on the held-out test split of the benchmark datasets used in this project. Results may differ on unseen projects, repositories, programming languages, or real-world development environments.

2. **Threshold selection:** The decision threshold of 0.430 was selected using the calibration set, with the final test set reserved for evaluation. The final test metrics are reported from `training_log_final.txt`.

3. **SMOTE-CV leakage (historical issue):** An earlier development version applied SMOTE before cross-validation, which could introduce synthetic-sample leakage into validation folds. The current training pipeline is intended to apply SMOTE only to training data. Verify the implementation in `train.py` before making stronger claims about leakage prevention.

4. **Not production-ready:** This pipeline is intended for research and portfolio demonstration. It has not been validated for production defect triage and should not be used for that purpose without further external validation.
