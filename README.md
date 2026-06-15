# fake_news_detection

Text classification to identify fake vs real news articles using NLP.

## Problem
With misinformation spreading fast, automating fake news detection is a 
practical NLP problem. This project builds a classifier trained on a labeled 
news dataset.

## Dataset
- Source: [Kaggle — ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Two CSV files: Fake.csv and True.csv
- Combined and labeled: 0 = Real, 1 = Fake

## Approach
- Dropped the `subject` column to avoid data leakage (subject alone can 
  perfectly separate classes, which wouldn't generalize to real-world data)
- Combined `title` + `text` into a single input field
- Used **TF-IDF vectorization** to convert text to features
- Trained a **Passive Aggressive Classifier**

## Results
- Accuracy: **99.73%**
- The model generalizes well because leakage sources were intentionally removed

## Tech Stack
Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
