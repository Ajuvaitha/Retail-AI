import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error

def forecast_demand(df, horizon=30):
    """Prophet if available, otherwise gradient boosting on calendar features."""
    try:
        from prophet import Prophet
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        m.fit(df)
        fut = m.make_future_dataframe(periods=horizon)
        fc = m.predict(fut)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        hist = fc.iloc[:-horizon]
        mae = mean_absolute_error(df.y, hist.yhat)
        return fc, mae, "Prophet"
    except Exception:
        def feats(d):
            return np.c_[d.dt.dayofweek, d.dt.dayofyear, (d - d.min()).dt.days]
        X, y = feats(df.ds), df.y.values
        g = GradientBoostingRegressor(random_state=0).fit(X, y)
        fut = pd.Series(pd.date_range(df.ds.min(), periods=len(df) + horizon))
        pred = g.predict(feats(fut))
        fc = pd.DataFrame({"ds": fut, "yhat": pred, "yhat_lower": pred * 0.9, "yhat_upper": pred * 1.1})
        return fc, mean_absolute_error(y, pred[:len(df)]), "GradientBoosting"

def segment_customers(df, k=4):
    X = StandardScaler().fit_transform(df[["recency_days", "frequency", "monetary"]])
    df = df.copy()
    df["segment"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    return df

def train_churn(df):
    cols = ["recency_days", "frequency", "monetary", "tenure_months"]
    Xtr, Xte, ytr, yte = train_test_split(df[cols], df.churned, test_size=0.25, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    return clf, auc, cols

def inventory_plan(inv, service_z=1.65):
    inv = inv.copy()
    inv["reorder_point"] = (inv.daily_demand * inv.lead_time_days
                            + service_z * inv.daily_demand * 0.3 * np.sqrt(inv.lead_time_days)).round()
    inv["status"] = np.where(inv.stock < inv.reorder_point, "Reorder", "OK")
    return inv

def drift_score(ref, new):
    """Population-stability style drift: mean shift in std units."""
    return float(abs(new.mean() - ref.mean()) / (ref.std() + 1e-9))
