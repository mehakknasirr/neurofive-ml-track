# 🚢 Titanic Survival Prediction App

An end-to-end Machine Learning web application built with Python and Streamlit that predicts passenger survival on the Titanic based on demographic and voyage details.

🔗 **Live Demo:** [Titanic Survival Predictor](https://neurofive-ml-track-dlwjdkvwwira8zz6xblenk.streamlit.app)

---

## 📌 Project Overview
This repository contains the complete Machine Learning workflow developed across the 5-week fellowship track. The project encompasses exploratory data analysis, feature engineering, model training and evaluation, Scikit-Learn pipeline creation, model serialization, and cloud deployment.

---

## 📅 Weekly Progression (Week 1 – Week 5)

### Week 1: Exploratory Data Analysis (EDA)
* Inspected dataset structure, data types, and missing value distributions.
* Handled missing data (`Age` median imputation, `Embarked` mode imputation, dropped high-null `Cabin` column).
* Conducted univariate and bivariate analysis to examine relationships between survival status and features like `Sex`, `Pclass`, and `Fare`.

### Week 2: Feature Engineering & Preprocessing
* Engineered new features: `FamilySize` (`SibSp` + `Parch` + 1) and binary indicator `IsAlone`.
* Applied One-Hot Encoding to categorical variables (`Sex`, `Embarked`).
* Scaled numerical values using `StandardScaler` to normalize distributions.

### Week 3: Model Training & Evaluation
* Trained multiple classification models: Logistic Regression, Decision Trees, Random Forest, and XGBoost.
* Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
* Selected Random Forest as the optimal baseline model based on overall accuracy and generalization performance.

### Week 4: Scikit-Learn Pipelines & Model Serialization
* Constructed a unified `ColumnTransformer` and `Pipeline` to prevent data leakage between training and test sets.
* Optimized model hyper-parameters using `GridSearchCV`.
* Serialized the complete fitted pipeline to `titanic_pipeline_model.pkl` using `joblib`.

### Week 5: Web Application & Cloud Deployment
* Developed an interactive user interface using `Streamlit` (`app.py`) for real-time model inference.
* Maintained dependency configuration in `requirements.txt` to guarantee environment reproducibility.
* Deployed the web application on **Streamlit Community Cloud**.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## 📂 Repository Structure

```text
├── app.py                         # Streamlit UI & inference script
├── titanic_pipeline_model.pkl      # Pre-trained Scikit-Learn pipeline
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
