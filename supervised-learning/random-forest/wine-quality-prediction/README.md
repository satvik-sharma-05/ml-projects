# Wine Quality Prediction using Random Forest Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Multi-class / Binary classification** project that predicts the **quality** of red wine (or white wine) based on physicochemical properties.

This notebook uses the **Random Forest Classifier** to classify wines as **good** or **bad** quality (binary version) with strong performance.

## 📌 Project Overview

Goal: Build a machine learning model that helps winemakers or consumers estimate wine quality from measurable features (acidity, sugar, alcohol, etc.).

Common approach:  
- Original dataset has quality scores 3–8 → many people convert to binary:  
  - Quality ≥ 7 → Good (1)  
  - Quality < 7 → Bad (0)

This notebook follows that binary classification approach.

## 📊 Dataset

- **Name**: Wine Quality Dataset (Red Wine variant most common)
- **Source**: UCI Machine Learning Repository / Kaggle
  - https://archive.ics.uci.edu/dataset/186/wine+quality
  - https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
- **Samples**: ~1,599 (red wine) / ~4,898 (white wine)
- **Features**: 11 physicochemical properties
- **Target**: Quality score (originally 0–10, here binarized: 0=bad, 1=good)

### Features

| Feature                | Description                              | Units          |
|------------------------|------------------------------------------|----------------|
| fixed acidity          | Fixed acidity                            | g(tartaric acid)/dm³ |
| volatile acidity       | Volatile acidity                         | g(acetic acid)/dm³ |
| citric acid            | Citric acid                              | g/dm³          |
| residual sugar         | Residual sugar                           | g/dm³          |
| chlorides              | Chlorides                                | g(sodium chloride)/dm³ |
| free sulfur dioxide    | Free sulfur dioxide                      | mg/dm³         |
| total sulfur dioxide   | Total sulfur dioxide                     | mg/dm³         |
| density                | Density                                  | g/cm³          |
| pH                     | pH                                       | —              |
| sulphates              | Sulphates                                | g(potassium sulphate)/dm³ |
| alcohol                | Alcohol                                  | % vol          |

Target (after binarization):  
- `1` → Good Quality  
- `0` → Bad Quality

## ⚙️ Tech Stack

- Python 3.8+
- NumPy, Pandas
- scikit-learn (RandomForestClassifier, train_test_split, accuracy_score)
- Jupyter Notebook

## 🛠️ Project Workflow

1. Load the dataset (`winequality-red.csv` or similar)
2. Exploratory Data Analysis (EDA)
   - Distribution of quality scores
   - Correlation heatmap
   - Boxplots for features vs quality
3. Binarize target: quality ≥ 7 → 1 (good), else 0 (bad)
4. Train-test split (usually 80/20)
5. Train Random Forest Classifier
6. Evaluate on test set
   - Accuracy
   - (Recommended: Precision, Recall, F1-score, Confusion matrix — since classes are imbalanced)
7. Build a prediction system for new wine samples

## 📈 Model Performance

With Random Forest (default or lightly tuned):

- **Test Accuracy**: ~0.90 – 0.93 (very good for this dataset)
- **F1-Score** (good class): usually ~0.65–0.75 (due to imbalance — fewer good wines)

> Note: Accuracy looks high because ~80–85% of wines are "bad" (quality < 7).  
> Always check confusion matrix & F1 for the minority class.

## 🔍 Prediction Example

```python
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)

input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data_as_numpy_array)

print("Good Quality Wine" if prediction[0] == 1 else "Bad Quality Wine")
# Output: Bad Quality Wine
```

## 🚀 How to Run

1. Clone / download the repository
2. Install dependencies
3. Place the dataset file (winequality-red.csv) in the project folder
4. Launch Jupyter Notebook
5. Open Wine_Quality_Prediction.ipynb and run all cells



## 👨‍💻 Author
Satvik
Chandigarh, India
GitHub: satvik-sharma-05