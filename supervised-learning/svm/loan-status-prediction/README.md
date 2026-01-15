# Loan Approval Prediction using Support Vector Machine (SVM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Binary classification** project that predicts whether a loan application will be **approved** (Y) or **rejected** (N) based on applicant information.

Built using **Support Vector Machine (SVM)** — a strong baseline for tabular classification tasks.

## 📌 Project Overview

This notebook implements an end-to-end machine learning pipeline to help banks/financial institutions automate loan approval decisions.

Goal: Learn supervised classification with SVM, feature encoding, handling categorical & numerical data, and model evaluation.

## 📊 Dataset

- **Name**: Loan Prediction Dataset (most common version from Kaggle / Analytics Vidhya)
- **Source**: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset or similar
- **Samples**: ~600–614 rows (small but realistic for learning)
- **Features**: 12 (mix of numeric & categorical)
- **Target**: `Loan_Status`  
  - `Y` → Loan Approved  
  - `N` → Loan Rejected

### Key Features

| Feature              | Description                              | Type       |
|----------------------|------------------------------------------|------------|
| Loan_ID              | Unique Loan ID                           | String     |
| Gender               | Male / Female                            | Categorical|
| Married              | Yes / No                                 | Categorical|
| Dependents           | 0 / 1 / 2 / 3+                           | Categorical|
| Education            | Graduate / Not Graduate                  | Categorical|
| Self_Employed        | Yes / No                                 | Categorical|
| ApplicantIncome      | Applicant income                         | Numeric    |
| CoapplicantIncome    | Co-applicant income                      | Numeric    |
| LoanAmount           | Loan amount requested (in thousands)     | Numeric    |
| Loan_Amount_Term     | Term of loan in months                   | Numeric    |
| Credit_History       | 1 = good, 0 = bad / no history           | Categorical|
| Property_Area        | Urban / Semiurban / Rural                | Categorical|

## ⚙️ Tech Stack

- Python 3.8+
- NumPy, Pandas
- Seaborn / Matplotlib (EDA & visualization)
- scikit-learn (SVM, train_test_split, accuracy_score, preprocessing)

## 🛠️ Project Workflow

1. Load the dataset (`train_u6lujuX_CVtuZ9i.csv` or similar)
2. Exploratory Data Analysis (EDA)
   - Missing value handling
   - Distribution of numerical features
   - Count plots for categorical features
   - Correlation heatmap
3. Data Preprocessing
   - Fill missing values (mode/median)
   - Label encoding for categorical variables
   - One-hot encoding (optional for better performance)
   - Feature scaling (important for SVM)
4. Train-test split (usually 80/20)
5. Train SVM classifier (`SVC(kernel='linear')` or `'rbf'`)
6. Evaluate model
   - Accuracy score
   - Confusion matrix
   - Precision, Recall, F1-score (especially important due to class imbalance)
7. Build a simple prediction function for new applicants

## 📈 Typical Performance

With basic preprocessing + linear SVM:

- Training Accuracy: ~78–82%
- Test Accuracy:     ~75–80%

> Credit_History is usually the most important feature (very strong predictor).  
> Using RBF kernel + scaling + proper encoding often pushes test accuracy to ~80–83%.

## 🔍 Prediction Example

```python
# Example input for a new applicant
new_applicant = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

# After preprocessing and scaling → model.predict(...)
prediction = model.predict(scaled_input)[0]
print("Loan Approved" if prediction == 1 else "Loan Rejected")
```
## 🚀 How to Run
1. Clone / download the repository
2. Install dependencies
3. Place the dataset file (train.csv or similar) in the project folder
4. Launch Jupyter Notebook


##  👨‍💻 Author
Satvik
Chandigarh, India
GitHub: satvik-sharma-05
