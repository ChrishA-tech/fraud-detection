"""End-to-end run: time-based split, baseline vs gradient boosting, threshold tuning,
label-delay experiment, drift check, latency check. Writes reports/metrics.json."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from .data import load_creditcard, make_synthetic
from .model import evaluate, make_model, pick_threshold
from .monitor import drift_report, latency_ms

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def time_split(df, train=0.70, val=0.15):
    """Fraud labels arrive late in real life, and models are always scored on the future.
    So we split by time, never randomly."""
    df = df.sort_values("Time").reset_index(drop=True)
    i, j = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def run(df, model_kind="xgboost", min_precision=0.80, seed=0):
    tr, va, te = time_split(df)
    feats = [c for c in df.columns if c not in ("Class", "Time")]
    out = {"rows": len(df), "fraud_rate": float(df["Class"].mean()), "models": {}}

    best = best_scores = None
    for kind in ("logreg", model_kind):
        m = make_model(kind, tr["Class"].to_numpy(), seed)
        m.fit(tr[feats], tr["Class"])
        thr = pick_threshold(va["Class"].to_numpy(), m.predict_proba(va[feats])[:, 1], min_precision)
        s_te = m.predict_proba(te[feats])[:, 1]
        out["models"][kind] = evaluate(te["Class"].to_numpy(), s_te, thr)
        if kind == model_kind:
            best, best_scores = m, s_te

    # Label-delay experiment: fraud labels often arrive weeks later, so the freshest
    # transactions cannot be used for training. Drop the most recent slice and re-score.
    out["label_delay"] = {}
    for frac in (0.0, 0.25, 0.5):
        sub = tr.iloc[: int(len(tr) * (1 - frac))]
        if sub["Class"].sum() < 5:
            continue
        m = make_model(model_kind, sub["Class"].to_numpy(), seed)
        m.fit(sub[feats], sub["Class"])
        thr = pick_threshold(va["Class"].to_numpy(), m.predict_proba(va[feats])[:, 1], min_precision)
        r = evaluate(te["Class"].to_numpy(), m.predict_proba(te[feats])[:, 1], thr)
        out["label_delay"][f"drop_latest_{int(frac * 100)}pct_of_train"] = {
            "pr_auc": r["pr_auc"], "precision": r["precision"], "recall": r["recall"]}

    out["drift_top_psi"] = drift_report(tr[feats], te[feats])
    out["latency"] = latency_ms(best, te[feats])

    REPORTS.mkdir(exist_ok=True)
    p, r, _ = precision_recall_curve(te["Class"], best_scores)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall (test, future data)")
    plt.tight_layout()
    plt.savefig(REPORTS / "pr_curve.png", dpi=120)
    plt.close()
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "hgb"])
    ap.add_argument("--min-precision", type=float, default=0.80)
    ap.add_argument("--synthetic", action="store_true", help="run offline on synthetic data")
    a = ap.parse_args()
    data = make_synthetic() if a.synthetic else load_creditcard()
    print(json.dumps(run(data, a.model, a.min_precision), indent=2))
