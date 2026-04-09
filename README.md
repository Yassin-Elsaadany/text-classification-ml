# 🧠 Text Classification ML – Spam Detection System

A professional machine learning project for **SMS spam detection** using classical NLP models, built with a clean and scalable architecture.

---

## 📌 Overview

This project implements and compares two powerful text classification models:

* 📈 Logistic Regression
* 📊 Multinomial Naive Bayes

The models are trained and evaluated on two datasets to analyze performance under **different data sizes**.

---

## 🎯 Objective

* Build a complete NLP pipeline
* Compare model performance across datasets
* Analyze behavior on **large vs small datasets**
* Produce reproducible and structured results

---

## 🗂️ Datasets

| Dataset | Size  | Ham   | Spam |
| ------- | ----- | ----- | ---- |
| SSCD1   | 5,574 | 4,827 | 747  |
| SSCD2   | 35    | 20    | 15   |

🔹 **SSCD1** → Large dataset (stable learning)
🔹 **SSCD2** → Small dataset (edge-case testing)

---

## ⚙️ Pipeline

### 1. Preprocessing

* Lowercasing
* Removing:

  * URLs
  * punctuation
  * special characters
* Cleaning whitespace

### 2. Feature Engineering

* TF-IDF Vectorization
* Max features: `5000`
* Stopwords removal

### 3. Data Splitting

* Train/Test: `80/20`
* Stratified split (when possible)

---

## 🧠 Models

### 🔹 Logistic Regression

* Strong for large datasets
* Learns weighted feature importance

### 🔹 Naive Bayes

* Fast & lightweight
* Performs well on small datasets

---

## 📊 Results

| Rank | Model               | Dataset | Accuracy   |
| ---- | ------------------- | ------- | ---------- |
| 🥇 1 | Logistic Regression | SSCD1   | **97.22%** |
| 🥈 2 | Naive Bayes         | SSCD1   | **96.95%** |
| 🥉 3 | Naive Bayes         | SSCD2   | **85.71%** |
| ❌ 4  | Logistic Regression | SSCD2   | **71.43%** |

---

## 📈 Key Insights

🔥 Large datasets → Logistic Regression dominates
🔥 Small datasets → Naive Bayes is more stable
🔥 Both models perform similarly when enough data exists

---

## 🧪 Example Prediction

```python
predict("Congratulations! You won a free iPhone 🎉")
# Output: Spam

predict("Hey, are we meeting today?")
# Output: Ham
```

---

## 📊 Model Comparison Visualization (Add Later)

> You can add graphs here:

* Accuracy comparison chart
* Confusion matrix
* Loss curves

---

## 🛠️ Installation

```bash
git clone https://github.com/Yassin-Elsaadany/text-classification-ml.git
cd text-classification-ml

pip install -r requirements.txt
```

---

## ▶️ Run Project

```bash
python src/training/train.py
python src/evaluation/evaluate.py
python src/inference/predict.py
```

---

## 🔄 Future Improvements

* [ ] Add Deep Learning (LSTM / Transformers)
* [ ] Add API (FastAPI)
* [ ] Deploy model (Docker / Cloud)
* [ ] Add real-time prediction UI
* [ ] Add precision, recall, F1-score

---

## 🏗️ Tech Stack

* Python 🐍
* Scikit-learn
* TF-IDF
* Jupyter Notebook

---

## 👨‍💻 Author

**Yassin Beshir**
Cybersecurity & AI Engineer

📧 [Y.Elsaada00673@student.aast.edu](mailto:Y.Elsaada00673@student.aast.edu)
🔗 https://github.com/Yassin-Elsaadany

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
