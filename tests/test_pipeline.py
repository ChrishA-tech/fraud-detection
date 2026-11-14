import numpy as np
from src.data import make_synthetic
from src.model import pick_threshold
from src.monitor import psi
from src.train import run, time_split


def test_time_split_has_no_leakage():
    tr, va, te = time_split(make_synthetic(5000))
    assert tr["Time"].max() <= va["Time"].min() <= va["Time"].max() <= te["Time"].min()


def test_threshold_respects_precision_target():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    s = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 5000), 0, 1)
    t = pick_threshold(y, s, min_precision=0.6)
    pred = s >= t
    assert (pred & (y == 1)).sum() / max(pred.sum(), 1) >= 0.6


def test_psi_detects_shift():
    rng = np.random.default_rng(2)
    assert psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.1
    assert psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)) > 0.25


def test_pipeline_beats_chance_on_synthetic():
    out = run(make_synthetic(20000, 0.02), model_kind="hgb")
    assert out["models"]["hgb"]["pr_auc"] > 0.3
    assert out["latency"]["within_budget"]
