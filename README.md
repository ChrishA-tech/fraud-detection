# Fraud Detection: imbalanced classification, threshold tuning and monitoring

A small, reproducible fraud-detection pipeline on the public ULB credit-card transactions
dataset (about 284k transactions, roughly 0.17% fraud). It is built around the problems fraud
teams face day to day, not just a model fit.

## What it covers

| Problem | How it is handled |
|---|---|
| **Extreme class imbalance** | Class weighting (`scale_pos_weight` for XGBoost) and PR-AUC as the main metric. Accuracy is not used. |
| **Evaluating on the future** | Time-ordered train / validation / test split, never random, so there is no leakage from later transactions. |
| **Analyst capacity** | The decision threshold is tuned on validation data for the highest recall at a minimum precision (default 80%), so alerts stay mostly correct. |
| **Delayed labels** | A label-delay experiment drops the newest slice of training data (fraud labels often arrive weeks late) and measures the effect on the future test set. |
| **Drift** | Population Stability Index (PSI) per feature between training and test periods. |
| **Real-time SLA** | Single-transaction scoring latency (p50 / p95 / p99) checked against a 300 ms budget. |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train                       # downloads the dataset via OpenML on first run
python -m src.train --model hgb           # use sklearn gradient boosting instead of XGBoost
python -m src.train --synthetic --model hgb   # offline smoke test on synthetic data
pytest                                    # unit tests
```

Outputs are written to `reports/metrics.json` and `reports/pr_curve.png`.

## Results

_Run the pipeline and paste your real numbers here._

| Model | PR-AUC | Precision | Recall | Alerts | False alarms |
|---|---|---|---|---|---|
| Logistic regression (baseline) | | | | | |
| XGBoost | | | | | |

- Label delay (PR-AUC when the newest 0% / 25% / 50% of training data is withheld): 
- Highest-drift features (PSI): 
- Scoring latency p99: 

## Design notes and limitations

- The features `V1`-`V28` are anonymised PCA components, so feature engineering and
  explainability are limited. A real system would add behavioural, device and network features.
- The label-delay experiment is a simulation (withholding recent training rows), not real
  delayed-label data.
- Latency is measured on one machine for model scoring only, without network or feature lookup time.
- Fraud patterns change. In production this would be paired with scheduled retraining and
  alerting on drift and on delayed ground truth.

## Structure

```
src/data.py      dataset loading and a synthetic generator
src/model.py     model factory, threshold selection, metrics
src/monitor.py   PSI drift and latency checks
src/train.py     end-to-end run
tests/           unit tests (time split leakage, threshold, PSI, pipeline)
```
