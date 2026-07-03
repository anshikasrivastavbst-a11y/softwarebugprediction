# Software Bug Prediction

A machine learning pipeline for predicting software defects using CK object-oriented
metrics and Halstead complexity metrics from 21 PROMISE and NASA MDP benchmark datasets.

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
  Threshold 0.475
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

Evaluated on a held-out 20% stratified test set (3,376 samples).

| Metric            | Value  |
|-------------------|--------|
| Macro F1          | 0.7896 |
| AUC-ROC           | 0.8494 |
| Buggy F1          | 0.7456 |
| Not-Buggy F1      | 0.8302 |
| Overall Accuracy  | 79.89% |
| Decision Threshold| 0.475  |

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

## 🚀 Live Demo
Software Bug Prediction System (ML Project)  
👉 https://softwarebugprediction.onrender.com


## Known Limitations

1. **Test-set threshold tuning:** The decision threshold (0.475) was selected by
   maximising Macro F1 on the same test set used for final evaluation. The correct
   approach is a separate calibration split (70/15/15). This makes the reported
   Macro F1 slightly optimistic. The AUC-ROC (0.8992) is threshold-independent
   and fully honest.

2. **CK-metrics ceiling:** Research literature (Menzies et al., D'Ambros et al.)
   documents that CK-metrics-only bug prediction typically achieves 80-85% Macro F1
   on PROMISE benchmark datasets. Results above this range on these datasets are
   generally indicative of overfitting or data leakage.

3. **SMOTE-CV leakage (fixed):** An earlier version applied SMOTE globally before
   the Optuna CV loop, allowing synthetic samples to leak into validation folds and
   inflate CV scores (0.84-0.85 CV vs 0.79 test). This was identified and fixed:
   SMOTE is now applied inside each fold on training indices only.

4. **Not production-ready:** This pipeline is intended for research and portfolio
   demonstration. It has not been validated for production defect triage and should
   not be used for that purpose without further external validation.
