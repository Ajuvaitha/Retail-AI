# Retail-AI
An AI-powered retail intelligence platform that helps analyze retail data, generate insights, and support smarter business decisions.

## RetailPulse
RetailPulse is a customer analytics and demand forecasting platform covering demand forecasting, customer segmentation, churn prediction, inventory optimization, and model monitoring.

## Features
- Demand forecasting (Prophet, with gradient-boosting fallback)
- RFM customer segmentation (K-Means)
- Churn prediction (Random Forest) with interactive predictor
- Inventory reorder-point planning
- Drift monitoring + MLflow experiment tracking
- Docker, Kubernetes manifests, GitHub Actions CI

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
## Docker
```
docker build -t retailpulse . && docker run -p 8501:8501 retailpulse
```
Data is synthetic (see data.py) and can be replaced with real sales/customer CSVs.
