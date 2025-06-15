# 🧠 Employee Attrition Prediction using Machine Learning

This project predicts whether an employee is likely to leave a company using machine learning techniques. The solution includes data preprocessing, model training, evaluation, and deployment via a Streamlit web app.

---

## 📂 Project Files

- `employee_attriation.ipynb` – Jupyter notebook for model development
- `Employee_Attrition.csv` – Dataset used for training/testing
- `app.py` – Streamlit app to interact with the model
- `best_model.pkl` – Saved ML model
- `feature_names.pkl` – Feature list used during training
- `requirements.txt` – Required Python packages
- `background.jpg` – Background image for Streamlit UI

---

## 📊 Dataset Overview

- **Source**: IBM HR Analytics dataset (or custom HR dataset)
- **Features**: Age, BusinessTravel, Department, DistanceFromHome, JobRole, MonthlyIncome, OverTime, etc.
- **Target**: `Attrition` (Yes or No)

---

## 🤖 Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest ✅ *(Best performing model)*
- Support Vector Machine (SVM)

The Random Forest model achieved the highest accuracy.

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix

> Note: Further evaluation using F1-score, precision, and recall is recommended for better insight into model performance on imbalanced data.

---

## 🚀 How to Run This Project

### 1. Clone the repository:
```bash

