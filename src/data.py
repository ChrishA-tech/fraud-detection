"""Data loading. Real data: the public ULB credit-card fraud dataset (via OpenML).
A synthetic generator is included so the pipeline and tests run offline."""
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"


def load_creditcard() -> pd.DataFrame:
    """Columns: Time, V1..V28, Amount, Class (1 = fraud). Cached locally after first download."""
    if CACHE.exists():
        return pd.read_csv(CACHE)
    from sklearn.datasets import fetch_openml
    df = fetch_openml("creditcard", version=1, as_frame=True, parser="auto").frame
    df.columns = [c.capitalize() if c.lower() in {"time", "amount", "class"} else c
                  for c in df.columns]
    df["Class"] = df["Class"].astype(int)
    df.to_csv(CACHE, index=False)
    return df


def make_synthetic(n: int = 20000, fraud_rate: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Small imbalanced dataset with the same schema, for tests and offline runs."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 10))
    X[y == 1, :3] += rng.normal(1.8, 0.6, size=(int(y.sum()), 3))  # fraud shifts a few features
    df = pd.DataFrame(X, columns=[f"V{i + 1}" for i in range(10)])
    df["Amount"] = np.where(y == 1, rng.lognormal(4.5, 1.0, n), rng.lognormal(3.5, 1.0, n))
    df["Time"] = np.sort(rng.uniform(0, 172800, n))  # two days, ordered
    df["Class"] = y
    return df
