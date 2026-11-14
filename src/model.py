"""Model factory and metrics. Fraud is rare, so we handle class imbalance explicitly and
report PR-AUC (not accuracy) plus precision/recall at a chosen threshold."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str, y_train, seed: int = 0):
    pos = max(int(np.sum(y_train)), 1)
    neg = len(y_train) - pos
    if kind == "logreg":  # baseline
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=1000))
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=neg / pos,  # imbalance handling
                                 eval_metric="aucpr", random_state=seed, n_jobs=-1)
        except ImportError:
            print("xgboost not installed, falling back to sklearn gradient boosting")
    return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=300,
                                          class_weight="balanced", random_state=seed)


def pick_threshold(y_true, scores, min_precision: float = 0.80) -> float:
    """Highest-recall threshold whose precision on validation data is >= min_precision.
    Models fraud ops reality: analysts can only review alerts that are mostly right."""
    p, r, t = precision_recall_curve(y_true, scores)
    ok = np.where(p[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return float(np.quantile(scores, 0.99))
    return float(t[ok[np.argmax(r[:-1][ok])]])


def evaluate(y_true, scores, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "alerts": int(pred.sum()),
        "fraud_caught": tp,
        "fraud_missed": fn,
        "false_alarms": fp,
        "fraud_rate_test": float(np.mean(y_true)),
    }
