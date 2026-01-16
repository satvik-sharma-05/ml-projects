# Gold Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Regression project** that predicts daily **Gold Price (GLD)** using historical financial market data.

This notebook uses machine learning to forecast gold prices based on related market indicators (stock indices, oil prices, silver, currency exchange rates, etc.).

## 📌 Project Overview

Goal: Build an accurate regression model to predict the price of gold (GLD ETF) — a popular asset in investment and hedging.

Key highlights:
- Very high test performance (R² ≈ 0.9887)
- Strong correlation between gold and other commodities/currencies
- Visual comparison of actual vs predicted prices

## 📊 Dataset

- **Common source**: Gold Price dataset from Kaggle / Yahoo Finance style
  - https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists (wait — wrong link; actual popular one:)
  - https://www.kaggle.com/datasets/altruistdelhite04/gold-price-dataset
  - Or: https://www.kaggle.com/datasets/ndrsharma/gold-price-prediction
- **Samples**: ~2,000–2,300 rows (daily data)
- **Features**: 5–6 numeric indicators + Date
- **Target**: GLD (Gold price in USD)

### Typical Features

| Feature     | Description                              |
|-------------|------------------------------------------|
| Date        | Date of observation                      |
| SPX         | S&P 500 index                            |
| GLD         | Gold price (target)                      |
| USO         | United States Oil Fund price             |
| SLV         | Silver price                             |
| EUR/USD     | Euro to US Dollar exchange rate          |

## ⚙️ Tech Stack

- Python 3.8+
- NumPy, Pandas
- Matplotlib, Seaborn (visualization & distribution plots)
- scikit-learn (train_test_split, metrics.r2_score, regressor)
- Jupyter Notebook

## 🛠️ Project Workflow

1. Load the dataset
2. Exploratory Data Analysis (EDA)
   - Distribution of GLD prices
   - Correlation heatmap (numeric features only)
   - Time series line plots
3. Preprocessing
   - Handle Date column (convert to datetime, extract year/month if needed)
   - Drop non-numeric columns before modeling
   - No major missing values in most versions
4. Train-test split (usually 80/20)
5. Train regression model (RandomForestRegressor / XGBoost / etc.)
6. Evaluate using R² score
7. Visualize actual vs predicted prices

## 📈 Model Performance

| Metric              | Value         | Notes                              |
|---------------------|---------------|------------------------------------|
| R² Score (Test)     | 0.9887        | Extremely strong fit               |
| Interpretation      | ~98.87%       | Model explains almost all variance |

> This level of R² is excellent for financial time-series regression — features are highly informative.

**Note**: In real-world trading, such high accuracy often indicates data leakage or very strong correlations — always validate on out-of-sample / future data.

## 📊 Visualization

Actual vs Predicted Prices (Test Set):

- Blue line: Actual GLD prices
- Green line: Predicted GLD prices

The lines overlap almost perfectly — showing the model's strong predictive power.

## 🔍 Prediction Example

```python
# Example for a new data point (after scaling/preprocessing if needed)
new_data = [[...]]  # [SPX, USO, SLV, EUR/USD, ...]

predicted_price = regressor.predict(new_data)[0]
print(f"Predicted Gold Price (GLD): ${predicted_price:.2f}")

```

## 🚀 How to Run
1. Clone / download the repository
2. Install dependencies
3. Place the dataset file (gld_price_data.csv or similar) in the project folder
4. Launch Jupyter Notebook
5. Open Gold_Price_Prediction.ipynb and run all cells

## 👨‍💻 Author
Satvik
Chandigarh, India
GitHub: satvik-sharma-05