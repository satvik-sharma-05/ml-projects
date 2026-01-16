# Heart Disease Prediction using Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Binary classification** project that predicts whether a person has **heart disease** (presence/absence) based on medical and demographic features.

This notebook uses **Logistic Regression** — a simple, interpretable, and widely used model for medical binary classification tasks.

## 📌 Project Overview

Goal: Build a model that can assist doctors or health apps in identifying individuals at risk of heart disease based on common clinical measurements.

This is a classic healthcare machine learning project — great for portfolios because it shows:
- Handling real medical data
- Binary classification with probabilistic interpretation
- Model interpretability (coefficients show feature importance)

## 📊 Dataset

- **Name**: Heart Disease UCI / Cleveland Dataset (most common version)
- **Source**: UCI Machine Learning Repository / Kaggle
  - https://archive.ics.uci.edu/dataset/45/heart+disease
  - https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Samples**: ~303 rows (small but realistic for learning)
- **Features**: 13 clinical & demographic predictors
- **Target**: `target` / `HeartDisease`
  - `0` → No heart disease
  - `1` → Heart disease present

### Features

| Feature          | Description                                      | Type       |
|------------------|--------------------------------------------------|------------|
| age              | Age in years                                     | Numeric    |
| sex              | Sex (1 = male, 0 = female)                       | Binary     |
| cp               | Chest pain type (0–3)                            | Categorical|
| trestbps         | Resting blood pressure (mm Hg)                   | Numeric    |
| chol             | Serum cholesterol (mg/dl)                        | Numeric    |
| fbs              | Fasting blood sugar > 120 mg/dl (1 = true)       | Binary     |
| restecg          | Resting electrocardiographic results (0–2)       | Categorical|
| thalach          | Maximum heart rate achieved                      | Numeric    |
| exang            | Exercise induced angina (1 = yes)                | Binary     |
| oldpeak          | ST depression induced by exercise                | Numeric    |
| slope            | Slope of peak exercise ST segment (0–2)          | Categorical|
| ca               | Number of major vessels colored by fluoroscopy  | Numeric    |
| thal             | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) | Categorical|

## ⚙️ Tech Stack

- Python 3.8+
- NumPy, Pandas
- scikit-learn (LogisticRegression, train_test_split, accuracy_score)
- Jupyter Notebook

## 🛠️ Project Workflow

1. Load the dataset (`heart.csv` or similar)
2. Exploratory Data Analysis (EDA)
   - Distribution of age, cholesterol, etc.
   - Count plots for categorical features vs target
   - Correlation heatmap
3. Preprocessing
   - Handle missing values (usually none in this dataset)
   - No scaling needed for Logistic Regression (but good practice)
4. Train-test split (80/20)
5. Train Logistic Regression model
6. Evaluate
   - Accuracy on train & test
   - (Recommended: Confusion matrix, Precision, Recall, F1 — important in medical tasks)
7. Build a prediction function for new patients

## 📈 Model Performance

| Set          | Accuracy       | Notes                              |
|--------------|----------------|------------------------------------|
| Training     | 0.8512 (85.12%)| Reasonable fit                     |
| Test         | 0.8197 (81.97%)| Good generalization, no strong overfitting |

> These are solid results for Logistic Regression on the UCI Heart Disease dataset.  
> Adding feature engineering or trying Random Forest / XGBoost often pushes test accuracy to 85–90%.

## 🔍 Prediction Example

```python
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data_as_numpy_array)

print("The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have a Heart Disease")
# Output: The Person does not have a Heart Disease

```

## 🚀 How to Run

1. Clone / download the repository
2. Install dependencies
3. Place the dataset file (heart.csv) in the project folder
4. Launch Jupyter Notebook
5. Open Heart_Disease_Prediction_LogisticRegression.ipynb and run all cells

## 👨‍💻 Author
Satvik
Chandigarh, India
GitHub: satvik-sharma-05