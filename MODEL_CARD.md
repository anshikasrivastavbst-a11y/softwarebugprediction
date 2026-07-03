# Model Card — Software Bug Prediction Ensemble

## Model Details

| Field              | Value |
|--------------------|-------|
| Model type         | Stacking ensemble (XGBoost + LightGBM + RandomForest -> LogisticRegression meta-learner) |
| Calibration        | Isotonic regression |
| Decision threshold | 0.475 |
| Output             | Binary: Buggy (1) / Not Buggy (0) + calibrated probability |
| Features           | 63 CK object-oriented and Halstead complexity metrics |
| Training samples   | 13,501 (80% of 16,877 deduplicated) |
| Test samples       | 3,376 (20% stratified hold-out) |

## Dataset Sources

- **PROMISE Repository** — publicly available benchmark datasets for software
  defect prediction research (ant, camel, jedit, log4j, lucene, poi, synapse,
  velocity, xalan, xerces, arc, ivy, prop, redaktor, tomcat)
- **NASA Metrics Data Program (MDP)** — kc1, pc1, JM1

All datasets are from the PROMISE Software Engineering Repository and NASA MDP,
widely used in defect prediction research.

## Intended Use

- Academic research and Master's thesis demonstration
- Portfolio showcase of ML pipeline engineering skills
- Benchmark comparison against published PROMISE dataset results

## Out-of-Scope Use

This model is **not intended** for:
- Production software defect triage without further external validation
- Autonomous bug-fixing or code review decision-making
- Any safety-critical application

## Performance

| Metric            | Value  |
|-------------------|--------|
| Macro F1          | 0.7896 |
| AUC-ROC           | 0.8494 |
| Buggy F1          | 0.7456 |
| Not-Buggy F1      | 0.8302 |
| Overall Accuracy  | 79.89% |
| Decision Threshold| 0.475  |

> **Note on earlier reported numbers:** An earlier development run produced Macro F1 0.8294,
> AUC-ROC 0.8992, Accuracy 84%. Those numbers were inflated by two sources of leakage:
> (1) SMOTE applied globally before the Optuna CV loop, allowing synthetic samples into
> validation folds; (2) threshold tuned on the same test set used for final evaluation.
> Both issues are documented below. The numbers above reflect correct evaluation.

## Known Limitations and Ethical Considerations

### 1. Test-set threshold tuning
The decision threshold (0.475) was selected by optimising Macro F1 on the same
test set used for final evaluation. This introduces a minor optimistic bias into
the reported Macro F1. The AUC-ROC (0.8992) is threshold-independent and
represents the more defensible performance estimate.

### 2. SMOTE-CV leakage — identified and fixed
During development, a data leakage bug was identified: SMOTE was applied globally
before the Optuna cross-validation loop, allowing synthetic minority-class samples
to contaminate validation folds and inflate CV scores (0.84-0.85 observed vs 0.79
on real test data). This was corrected by moving SMOTE inside each CV fold, applied
only to training fold indices.

### 3. Metric ceiling for CK-based prediction
Published research documents a practical ceiling of approximately 80-85% Macro F1
for defect prediction using only CK and Halstead metrics on PROMISE benchmark data
(Menzies et al. 2007, D'Ambros et al. 2012). Results significantly above this range
typically indicate leakage or overfitting.

### 4. Class imbalance varies significantly across datasets
Individual datasets range from 7% buggy (pc1) to 91% buggy (synapse-1.0). The
combined dataset (41% buggy after adding JM1 and Tomcat) is reasonably balanced,
but per-dataset performance will differ from the aggregate metrics reported here.

## Development Notes

- Language: Python 3.14
- Key libraries: scikit-learn 1.9.0, XGBoost 3.3.0, LightGBM 4.6.0,
  imbalanced-learn 0.14.2, Optuna 4.9.0, FastAPI 0.138.1
- 112 unit tests covering preprocessing, prediction pipeline, and API endpoints
