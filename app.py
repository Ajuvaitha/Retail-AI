import streamlit as st, plotly.express as px, pandas as pd
from data import make_sales, make_customers, make_inventory
from models import *

st.set_page_config(page_title="RetailPulse", page_icon="📈", layout="wide")
st.title("📈 RetailPulse")
st.caption("AI-Powered Customer Analytics & Demand Forecasting Platform")

@st.cache_data
def load():
    return make_sales(), make_customers(), make_inventory()
sales, cust, inv = load()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Demand Forecast", "Segmentation & Churn", "Inventory", "Monitoring"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total units sold", f"{int(sales.y.sum()):,}")
    c2.metric("Customers", f"{len(cust):,}")
    c3.metric("Churn rate", f"{cust.churned.mean():.1%}")
    c4.metric("Avg customer spend", f"₹{cust.monetary.mean():,.0f}")
    st.plotly_chart(px.line(sales, x="ds", y="y", title="Daily sales"), use_container_width=True)

with tab2:
    h = st.slider("Forecast horizon (days)", 7, 90, 30)
    fc, mae, name = forecast_demand(sales, h)
    st.write(f"Model: **{name}** · In-sample MAE: **{mae:.1f}** units")
    fig = px.line(fc, x="ds", y=["yhat", "yhat_lower", "yhat_upper"], title="Demand forecast")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    seg = segment_customers(cust)
    st.plotly_chart(px.scatter(seg, x="frequency", y="monetary", color=seg.segment.astype(str),
                    title="RFM customer segments"), use_container_width=True)
    st.dataframe(seg.groupby("segment")[["recency_days", "frequency", "monetary"]].mean().round(1))
    clf, auc, cols = train_churn(cust)
    st.write(f"Churn model (Random Forest) ROC-AUC: **{auc:.2f}**")
    st.subheader("Predict churn for a customer")
    r = st.number_input("Recency (days)", 1, 365, 120)
    f = st.number_input("Frequency", 1, 50, 5)
    m = st.number_input("Monetary", 0.0, 5000.0, 300.0)
    t = st.number_input("Tenure (months)", 1, 60, 12)
    p = clf.predict_proba([[r, f, m, t]])[0, 1]
    st.metric("Churn probability", f"{p:.1%}")

with tab4:
    plan = inventory_plan(inv)
    st.write(f"SKUs needing reorder: **{(plan.status == 'Reorder').sum()}** of {len(plan)}")
    st.dataframe(plan.sort_values("status"), use_container_width=True)

with tab5:
    ref, new = sales.y[:365], sales.y[365:]
    d = drift_score(ref, new)
    st.metric("Sales drift (std units, year1→year2)", f"{d:.2f}")
    st.success("Stable") if d < 0.5 else st.warning("Drift detected — retrain recommended")
    try:
        import mlflow
        mlflow.set_experiment("retailpulse")
        with mlflow.start_run():
            mlflow.log_metric("churn_auc", auc); mlflow.log_metric("forecast_mae", mae)
        st.caption("Metrics logged to MLflow.")
    except Exception:
        st.caption("MLflow logging unavailable in this environment.")
