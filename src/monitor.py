"""Monitoring helpers: feature drift (PSI) and latency against a real-time budget."""
import time
import numpy as np


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(X_ref, X_new, top: int = 10) -> dict:
    scores = {c: psi(X_ref[c].to_numpy(), X_new[c].to_numpy()) for c in X_ref.columns}
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top])


def latency_ms(model, X, n: int = 500) -> dict:
    """Single-transaction scoring latency. Real-time fraud APIs often have ~300 ms budgets."""
    rows = [X.iloc[[i]] for i in range(min(n, len(X)))]
    model.predict_proba(rows[0])  # warm-up
    times = []
    for r in rows:
        t0 = time.perf_counter()
        model.predict_proba(r)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "budget_ms": 300.0,
            "within_budget": bool(np.percentile(times, 99) < 300.0)}
