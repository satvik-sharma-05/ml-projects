# Diabetes Prediction using Support Vector Machine (SVM)

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**Binary classification** project that predicts whether a person has diabetes based on diagnostic medical measurements using a **Support Vector Machine (SVM)**.

## 📌 Project Overview

This notebook builds a simple yet effective **SVM classifier** (with linear kernel) to predict diabetes (yes/no) using the well-known **PIMA Indians Diabetes Dataset**.

Goal: Learn fundamental supervised learning concepts including preprocessing, scaling, model training, evaluation, and building a basic prediction interface.

## 📊 Dataset

- **Name**: PIMA Indians Diabetes Database  
- **Source**: UCI Machine Learning Repository / Kaggle  
- **Instances**: 768  
- **Features**: 8 numeric medical predictors  
- **Target**: `Outcome` (0 = non-diabetic, 1 = diabetic)  
- **Class distribution**: ~500 non-diabetic / ~268 diabetic (imbalanced)

### Features

| Feature                    | Description                                      | Units          |
|----------------------------|--------------------------------------------------|----------------|
| Pregnancies                | Number of times pregnant                         | —              |
| Glucose                    | Plasma glucose concentration (2-hour OGTT)       | mg/dL          |
| BloodPressure              | Diastolic blood pressure                         | mm Hg          |
| SkinThickness              | Triceps skin fold thickness                      | mm             |
| Insulin                    | 2-hour serum insulin                             | μU/mL          |
| BMI                        | Body mass index                                  | kg/m²          |
| DiabetesPedigreeFunction   | Diabetes pedigree function (genetic score)       | —              |
| Age                        | Age                                              | years          |

> **Note**: Several features contain invalid 0 values (e.g. Glucose, BMI, Insulin) that should be treated as missing in real analysis.

## ⚙️ Technologies Used

- Python 3.x  
- Jupyter Notebook  
- NumPy  
- Pandas  
- scikit-learn (StandardScaler, SVC, train_test_split, accuracy_score)  
- matplotlib / seaborn (optional visualizations)

## 🛠️ Workflow

1. Load and explore the dataset (`diabetes.csv`)
2. Handle data inspection & basic cleaning
3. Separate features (X) and target (y)
4. Feature scaling using `StandardScaler` (very important for SVM)
5. Train / test split (usually 80:20 or 75:25)
6. Train SVM model (`SVC(kernel='linear')`)
7. Evaluate on training and test sets (mainly accuracy)
8. Build a simple prediction function for new patient data

## 📈 Typical Performance

With a basic linear SVM + standard preprocessing you can usually expect:

- Training accuracy : **~76–80%**  
- Test accuracy     : **~75–78%**  

> These values are typical for linear SVM on this dataset without advanced feature engineering, imbalance handling (SMOTE), or hyperparameter tuning.  
> RBF kernel or tuned parameters often reach **~80–83%**.

## 🚀 How to Run

1. Clone / download the repository

```bash
git clone <your-repo-url>
cd supervised-learning/svm/diabetes-prediction
```

2. Install dependencies (preferably in a virtual environment)
   ```bash
   pip install -r requirements.txt
# or directly:
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

3. Launch Jupyter Notebook
```bash
jupyter notebook
```

4. Open Diabetes_Prediction_SVM.ipynb and run all cells

   ## 🔮 Example Prediction
   # Example input: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
input_data = [1, 89, 66, 23, 94, 28.1, 0.167, 21]

prediction = predict_diabetes(input_data)
print("The person is diabetic." if prediction[0] == 1 else "The person is NOT diabetic.")



## 📚 Key Learnings
Why feature scaling is critical for distance-based algorithms like SVM
Basic usage of SVC with linear vs RBF kernel
Train/test split and avoiding data leakage
Limitations of accuracy on imbalanced classes
Building a minimal end-to-end prediction function


## 👨‍💻 Author
Satvik Sharma
Chandigarh, India
GitHub: satvik-sharma-05


