# Pharmaceutical Demand Forecasting & Anomaly Detection

A machine learning pipeline for forecasting weekly pharmaceutical drug demand and detecting anomalous sales patterns — built as the ML component of a full-stack pharmacy management system.

> **Project context:** This ML module was integrated into a pharmacy management platform built with MERN stack and blockchain by a team. This repository contains only my contribution — the forecasting and anomaly detection model.

---

## What This Does

Pharmacies lose money in two ways — overstocking drugs that expire, or running out of stock and losing sales. This model addresses both:

- **Demand Forecasting** — predicts weekly sales for 8 drug categories up to 30 days ahead
- **Anomaly Detection** — flags weeks where sales deviate unusually from learned normal patterns, which can indicate supply disruptions, data errors, or suspicious bulk orders

---

## Model Architecture

Three algorithms combined, each capturing different patterns:

| Model | What it captures |
|---|---|
| **Prophet** (Meta) | Long-term trends and seasonal patterns |
| **LSTM** | Sequential memory — how recent weeks influence the next |
| **XGBoost** | Non-linear relationships between lag features and date features |

Outputs are blended using a **Ridge regression meta-learner (stacking)** — trained on a validation slice to learn optimal data-driven weights instead of hardcoding them.

```
Raw Sales Data
      │
      ├──► Prophet ───────────────────────┐
      │                                   │
      ├──► LSTM (sequence window = 12) ───┼──► Ridge Meta-Learner ──► Final Forecast
      │                                   │
      └──► XGBoost (lag + date features) ─┘
```

---

## Results

| Model | MAE |
|---|---|
| Prophet | 58.54 |
| LSTM | 45.97 |
| XGBoost | 61.65 |
| Old fixed weights (0.4 / 0.4 / 0.2) | 36.21 |
| **Final Ensemble — meta-learner** | **31.98** |
| **Improvement over best single model** | **30.4% better than LSTM** |

MAE = Mean Absolute Error. Lower is better.

---

## Anomaly Detection

Uses **Isolation Forest** on 5 features simultaneously:

- Raw sales value
- 4-week rolling mean
- 4-week rolling standard deviation
- Lag-1 (previous week)
- Lag-2 (two weeks prior)

More robust than simple threshold methods because it learns what normal looks like across multiple dimensions, not just a single value.

---

## Dataset

- **Source:** [Pharma Sales Data — Milan Zdravković](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)
- **Contents:** Weekly sales of 8 drug categories from a pharmacy (2014–2019)
- **Drug categories:** M01AB (Diclofenac), M01AE (Ibuprofen), N02BA (Aspirin), N02BE (Paracetamol), N05B (Diazepam), N05C (Zolpidem), R03 (Salbutamol), R06 (Cetirizine)
- **Note:** Individual medicine-level data was unavailable. Each category was mapped to its primary representative drug and forecasting is done at category level.
- **Preprocessing:** Weekly aggregation, rolling smoothing (window=4), log transform, 80/20 train-test split

> The dataset file is not included in this repository. Download `salesweekly.csv` from the Kaggle link above and place it in the same folder as the notebook before running.

---

## Key Design Decisions

**Why three models instead of one?**
Each model sees the data differently. Prophet misses short-term sequence patterns. LSTM needs lots of data to beat simpler models. XGBoost does not capture trend natively. Combining them covers each model's blind spots.

**Why Ridge for the meta-learner?**
Simple, interpretable, and with `positive=True` keeps weights non-negative. Regularisation prevents overfitting on the small validation set.

**Why Isolation Forest?**
Does not assume a particular distribution and uses multiple features simultaneously. More defensible than threshold-based methods for multivariate anomaly detection.

---

## Honest Limitations

- Per-medicine forecasts distribute total predicted sales by historical ratio — not individually trained per drug
- LSTM trained for 10 epochs with basic architecture — hyperparameter tuning would improve results
- Dataset is from a single European pharmacy — generalisation to other regions untested
- Anomaly detection is unsupervised — no ground truth labels to validate against

---

## Tech Stack

Python • TensorFlow/Keras • scikit-learn • XGBoost • Prophet • SHAP • Pandas • NumPy • Matplotlib

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet tensorflow shap

jupyter notebook Prophet_LSTM_XGBoost_Ensemble_Final.ipynb
```

Run all cells top to bottom. The final cell prints a clean summary of all MAE scores.

---

## Project Structure

```
├── Prophet_LSTM_XGBoost_Ensemble_Final.ipynb   # Main notebook
├── .gitignore
├── LICENSE
└── README.md
```

---

## Related Work

This ML module was integrated into a full-stack pharmacy management system with counterfeit detection features, built with MERN stack and blockchain. The model outputs structured JSON consumed by the backend API.
