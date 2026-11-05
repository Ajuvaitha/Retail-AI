import numpy as np, pandas as pd

def make_sales(seed=42, days=730):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days)
    t = np.arange(days)
    base = 200 + 0.15 * t + 40 * np.sin(2 * np.pi * t / 7) + 30 * np.sin(2 * np.pi * t / 365)
    sales = np.maximum(base + rng.normal(0, 15, days), 0).round()
    return pd.DataFrame({"ds": dates, "y": sales})

def make_customers(seed=7, n=1500):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "recency_days": rng.integers(1, 365, n),
        "frequency": rng.poisson(8, n) + 1,
        "monetary": rng.gamma(2.5, 120, n).round(2),
        "tenure_months": rng.integers(1, 60, n),
    })
    z = 0.012 * df.recency_days - 0.15 * df.frequency - 0.002 * df.monetary - 0.01 * df.tenure_months + 0.5
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df

def make_inventory(seed=11, n=40):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sku": [f"SKU-{i:03d}" for i in range(1, n + 1)],
        "stock": rng.integers(0, 400, n),
        "daily_demand": rng.uniform(5, 60, n).round(1),
        "lead_time_days": rng.integers(2, 14, n),
    })
