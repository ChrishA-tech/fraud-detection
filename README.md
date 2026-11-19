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

Measured on the full ULB dataset (284,807 transactions, 492 frauds, 0.17% fraud). Time-ordered split 70 / 15 / 15:
the test set is the latest 15% of transactions (42,722 rows, 52 frauds). The decision threshold was picked on the
validation set for the highest recall at precision ≥ 0.80, then applied unchanged to the test set.

| Model | PR-AUC | ROC-AUC | Precision | Recall | Alerts | Fraud caught | False alarms |
|---|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.708 | 0.977 | 0.844 | 0.731 | 45 | 38 / 52 | 7 |
| XGBoost (class-weighted) | **0.762** | 0.974 | 0.696 | 0.750 | 56 | 39 / 52 | 17 |

- **XGBoost ranks fraud better** (PR-AUC 0.762 vs 0.708), but at the threshold chosen on validation its test precision
  fell to 0.70, below the 0.80 target: the threshold did not transfer cleanly to later data. Logistic regression held
  its precision (0.84). With only 52 test frauds, one or two cases move these numbers a lot.
- **Label delay** (newest training data withheld, as if fraud labels arrive late): PR-AUC 0.762 with 0% withheld,
  0.760 with 25%, 0.747 with 50%. Losing the most recent half of the training data costs about 0.015 PR-AUC.
- **Highest-drift features (PSI, train vs test period):** V1 1.01, V3 0.75, V28 0.53, V11 0.34. Above 0.25 is usually
  treated as a major shift, so these are the features to monitor first.
- **Scoring latency** (single transaction, model only): p50 1.4 ms, p95 1.7 ms, p99 2.3 ms, well inside the 300 ms budget.

Reproduce: `python -m src.train` (writes `reports/metrics.json` and `reports/pr_curve.png`).

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
