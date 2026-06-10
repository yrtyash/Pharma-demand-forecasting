# Pharma Demand Forecasting

I built this as the ML component of a larger pharmacy management system my team was working on. The backend (MERN + blockchain) was handled by my teammate — my part was figuring out how to predict drug demand reliably and flag unusual sales patterns.

The core problem: pharmacies either overstock and waste money on expired drugs, or run out of stock and lose sales. Both are avoidable with decent forecasting.

---

## What I built

A forecasting pipeline that combines three models and blends their outputs, plus an anomaly detector that flags weeks where sales behaved unusually.

**Why three models?**
I tried Prophet alone first — it handles seasonality well but misses short-term patterns. LSTM captures sequential patterns but needs a lot of data to outperform simpler models. XGBoost with lag features picks up non-linear relationships the others miss. Each one has blind spots the others cover, so combining them made sense.

**The ensemble part**
My first version just used fixed weights which I picked by hand — that was a bad idea since there was no principled reason for any particular combination. Replaced it with a Ridge regression meta-learner that learns the optimal blend from validation data. This is called stacking and it's a standard ensemble technique. The learned weights came out roughly equal across all three models (LSTM: 0.329, Prophet: 0.346, XGBoost: 0.325), which actually makes sense — each model captures something different so none should dominate.

**Anomaly detection**
Used Isolation Forest on 5 features (sales value, rolling mean, rolling std, lag-1, lag-2). Works better than a simple threshold because it learns what normal looks like across multiple dimensions rather than just flagging anything above 2 standard deviations.

---

## Results

| Model | MAE |
|---|---|
| Prophet | 58.54 |
| LSTM | 45.97 |
| XGBoost | 61.65 |
| **Final Ensemble (meta-learner)** | **22.97** |

The meta-learner ensemble gave a 50% improvement over the best individual model (LSTM). MAE = Mean Absolute Error, lower is better.

---

## Dataset

Pharma sales data from Kaggle — weekly sales across 8 drug categories from a single pharmacy.
Link: https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data

Download `salesweekly.csv` and put it in the same folder as the notebook before running.

One thing to note: individual medicine-level data wasn't available so I mapped each drug category to its main representative medicine (M01AB to Diclofenac, N02BE to Paracetamol, etc.) and forecasted at category level. Per-medicine forecasts in the dashboard output distribute total predicted sales by historical ratio — not ideal but practical given the data available.

---

## Honest limitations

- LSTM is only 10 epochs with a basic architecture, no hyperparameter tuning
- Per-medicine forecasts use ratio distribution not individual models
- Data is from one European pharmacy so generalisation is uncertain
- Anomaly detection is unsupervised so there's no ground truth to validate against

---

## Stack

Python, TensorFlow/Keras, scikit-learn, XGBoost, Prophet, SHAP, Pandas, NumPy, Matplotlib

---

## How to run

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet tensorflow shap
jupyter notebook "Model Final.ipynb"
```

Run all cells top to bottom. The last cell prints a summary of all MAE scores.

---

## Context

This model was integrated into a full-stack pharmacy management system with counterfeit detection features. The model outputs structured JSON that the MERN backend consumes to show forecasts and inventory suggestions on the dashboard.
