# 🧠 UNIFIED MENTOR PRIVATE LIMITED INTERNSHIP PROJECTS

Welcome to the official internship project repository! This repo contains 5 different machine learning projects built during the internship using Python, Streamlit, and advanced ML techniques.

---

## ⚙️ Setup Instructions

1. **Create a virtual environment using `uv`**:
   ```bash
   uv venv 
   source .venv/bin/activate
   ```

2. **Install dependencies** (if any are present in a `requirements.txt` or via Streamlit, scikit-learn, etc.)

---

## 📁 Project Structure

```
Unified_Mentor_Internship_Projects/
│
├── Animal_Classifications/
├── fraud_detection/
├── Heart_Disease/
├── Mobile_Phone_Pricing/
└── Vehicle_Price_Prediction/
```

---

## 🐾 Animal Classifications
- A Streamlit-based image classification system to identify animals.
- Run the app using:
  ```bash
  streamlit run new_app.py
  ```
- The `new_app.py` utilizes `InferenceHTTPClient` with `api_url` and `api_key` for real-time inference.

---

## ❤️ Heart Disease Prediction
- Built with `ExtraTreesClassifier`, achieving an accuracy of **84.6%**.
- Limited dataset, but robust performance.
- Run the app using:
  ```bash
  streamlit run app.py
  ```

---

## 📱 Mobile Phone Price Prediction
- Built using **Support Vector Classifier (SVC)**.
- Achieved **97.5%** accuracy.
- Run with Streamlit:
  ```bash
  streamlit run app.py
  ```

---

## 🕵️ Fraud Detection

### ✅ Conclusion:
📌 **Best Choice: CatBoost**

- Excellent balance of **Precision (0.56)** and **Recall (0.92)**.
- Highest **PR AUC (0.9325)**, ideal for imbalanced fraud datasets.
- Effective at minimizing **false negatives** with strong performance overall.

🔹 **Alternative Option:**  
If your goal is **maximum fraud catch rate (high recall)**, consider **XGBoost**.

---

## 🚗 Vehicle Price Prediction

### Model Performance Summary:

Based on metrics like **R², RMSE, and MAE**, the **Decision Tree Regressor** performed the best.

### 📊 Why is Decision Tree the Best?
✅ **Highest R² Score (0.875)** → Explains most variance in data  
✅ **Lowest RMSE (10.67)** → Lowest prediction error  
✅ **Lowest MAE (7.55)** → More accurate predictions on average

---

## 🔬 Feature Engineering and EDA

Across all projects, proper **exploratory data analysis (EDA)** and **feature engineering** techniques were used:
- Feature Selection
- Feature Transformation
- Handling Multicollinearity
- Outlier Detection
- Normalization/Scaling
- Model Evaluation

---

## 📌 Final Note

These projects highlight real-world applications of ML with a strong emphasis on **model interpretability**, **performance**, and **deployment** through **Streamlit** and **API integrations**.

---
## 🛡️ License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and share with attribution.

---
## 📢 Connect with Me

Let's collaborate! Connect with me on:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/v-rithul-06b5632b6/)  

🚀 **Happy Coding!**
