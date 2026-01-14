# Fake News Detection using Logistic Regression & TF-IDF

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-≥3.8-green?logo=nltk&logoColor=white)](https://www.nltk.org/)

**Binary text classification** project that identifies whether a news article is **Fake** or **Real** using Natural Language Processing (NLP) techniques and Logistic Regression.

## 📌 Project Overview

This notebook builds a complete end-to-end pipeline to classify news articles as fake (1) or real (0) based only on their title and/or text content.

Key techniques demonstrated:
- Text preprocessing (cleaning, stemming, stopword removal)
- Feature extraction with **TF-IDF Vectorizer**
- Training a simple yet effective **Logistic Regression** classifier
- Building a prediction function for new articles

## 📊 Dataset

- **Source**: Fake and Real News Dataset (Kaggle) or similar widely-used collections
- **Total articles**: ~40,000–45,000 (balanced or near-balanced)
- **Classes**:
  - `0` → Real News
  - `1` → Fake News
- **Main columns used**:
  - `title` (headline)
  - `text` (article body)
  - `label` / `class` (target)

Popular Kaggle sources:
- https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection

## ⚙️ Tech Stack

- Python 3.8+
- NumPy, Pandas
- NLTK (text preprocessing & stopwords)
- scikit-learn (TF-IDF, LogisticRegression, train_test_split, metrics)
- Jupyter Notebook

## 🛠️ Project Workflow

1. Load and combine real/fake news datasets
2. Create unified DataFrame + label column
3. Text preprocessing pipeline:
   - Convert to lowercase
   - Remove punctuation, numbers, special characters
   - Tokenization
   - Remove stopwords
   - Porter Stemming / Snowball Stemming
4. Combine title + text (or use one)
5. TF-IDF Vectorization (with n-grams, min_df, max_features tuning)
6. Train-test split (usually 80/20 or 75/25)
7. Train Logistic Regression model
8. Evaluate:
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion matrix
9. Build simple prediction function for new articles

## 📈 Typical Performance

With basic TF-IDF + Logistic Regression you can expect:

- Training Accuracy: ~98–99%
- Test Accuracy:     ~97–98.5%
- F1-Score (macro):  ~0.97–0.98

> Very strong results are common on clean, well-labeled fake/real news datasets.

## 🔍 Prediction Example

```python
def predict_news(text):
    processed = preprocess(text)                # your preprocessing function
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return "Fake News" if prediction == 1 else "Real News"

# Usage
article = "Breaking: Aliens confirmed living among us..."
print(predict_news(article))
# Output: Fake News

## 👨‍💻 Author
Satvik
Chandigarh, India
GitHub: satvik-sharma-05
