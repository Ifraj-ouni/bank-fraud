# 🏦 Bank Fraud Detection

A machine learning project to detect fraudulent bank transactions using XGBoost, with an interactive web application built with Streamlit.

---

## 📌 Project Overview

This project aims to identify fraudulent transactions from bank data using supervised machine learning techniques. It covers the full data science pipeline: data cleaning, exploratory data analysis (EDA), feature engineering, model training, and deployment via a web app.

---

## 📁 Project Structure

```
bank-fraud/
│
├── EDAandClean.ipynb        # Exploratory Data Analysis & Data Cleaning
├── Prepmodel.ipynb          # Feature Engineering & Model Training
├── app.py                   # Streamlit Web Application
├── xgb_modelFraud.pkl       # Trained XGBoost Model
├── .gitignore
└── README.md
```

> ⚠️ **Note:** Large CSV data files are not included in this repository due to GitHub's file size limits. See the [Data](#-data) section below.

---

## ⚙️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Preprocessing & evaluation |
| XGBoost | Fraud classification model |
| Streamlit | Web application |
| Jupyter Notebook | EDA & model development |

---

## 🔄 Pipeline

1. **Data Cleaning** (`EDAandClean.ipynb`)
   - Handle missing values
   - Remove duplicates
   - Feature selection and encoding

2. **Model Preparation** (`Prepmodel.ipynb`)
   - Train/test split
   - Handle class imbalance
   - Train XGBoost classifier
   - Evaluate with accuracy, precision, recall, F1-score

3. **Web App** (`app.py`)
   - Input transaction features
   - Predict fraud in real-time using the saved model

---

## 💾 Data

The dataset used contains bank transaction records with features such as transaction amount, type, account balances, etc.

Due to GitHub file size limitations, the data files are not included in this repository.

> 📥 You can download the dataset from: *(add your Google Drive / Kaggle link here)*

Expected data files:
- `Dataraw/dataPS.csv` — Raw dataset
- `dataClean/dataClean.csv` — Cleaned dataset
- `X_test.csv` / `y_test.csv` — Test sets

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Ifraj-ouni/bank-fraud.git
cd bank-fraud
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib
```

### 3. Run the web app
```bash
streamlit run app.py
```

## 👨‍💻 Author

**Ifraj Ouni // Yessmine Hassad**  
Student at ISTIC  
[GitHub](https://github.com/Ifraj-ouni)

---

## 📄 License

This project is for educational purposes only.
