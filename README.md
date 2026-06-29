# 🧠 DemandIQ - Retail Demand Forecasting & Inventory Optimization System

DemandIQ is an end-to-end machine learning application that helps retailers forecast product demand and make smarter inventory decisions. The system predicts future demand using historical sales and business-related factors such as pricing, promotions, seasonality, weather conditions, inventory levels, and competitor pricing.



## 📌 Problem Statement

Retail businesses frequently face challenges such as:

- Overstocking products
- Inventory shortages
- Revenue loss due to inaccurate demand planning
- Difficulty understanding demand drivers

DemandIQ addresses these challenges using machine learning and explainable AI to forecast demand and provide actionable inventory insights.

---

## ✨ Features

- 📈 Demand Forecasting using XGBoost
- 📦 Inventory Risk Assessment
- 🚨 Inventory Shortage Detection
- 📊 KPI Dashboard
- 🔍 SHAP Explainability
- 🏆 Demand Driver Analysis
- 📉 Inventory vs Forecast Visualization
- 🌐 Streamlit Web Application

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Scikit-learn
- XGBoost

### Data Processing
- Pandas
- NumPy

### Explainable AI
- SHAP

### Visualization
- Plotly

### Deployment
- Streamlit

---

## 📂 Dataset

Source:
Retail Store Inventory and Demand Forecasting Dataset (Kaggle)

Dataset Characteristics:
- Synthetic retail dataset
- Multiple stores and product categories
- Inventory, pricing, promotional, and seasonal information
- Weather and competitor pricing features
- Demand forecasting target variable

Dataset Link:

https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting

---

## 🔄 Workflow

```text
Raw Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Label Encoding
    ↓
XGBoost Training
    ↓
5-Fold Cross Validation
    ↓
SHAP Explainability
    ↓
Streamlit Dashboard
```

---

## 📊 Model Performance

### Model Comparison

| Model | Test R² Score |
|---------|---------|
| Random Forest | 0.776 |
| XGBoost | 0.865 |

### XGBoost Cross Validation Results

| Fold | R² Score |
|------|------|
| Fold 1 | 0.888 |
| Fold 2 | 0.909 |
| Fold 3 | 0.886 |
| Fold 4 | 0.910 |
| Fold 5 | 0.876 |

### Average 5-Fold Cross Validation Score

**R² = 0.894**

The XGBoost model demonstrated the best performance and was selected for deployment.

---

## 📈 Input Features

The model uses multiple business and environmental variables:

- Store ID
- Product ID
- Category
- Region
- Inventory Level
- Price
- Discount
- Weather Condition
- Promotion
- Competitor Pricing
- Seasonality
- Epidemic Indicator
- Year
- Month
- Day Of Week
- Quarter

---

## 🧠 Explainable AI with SHAP

DemandIQ integrates SHAP (SHapley Additive exPlanations) to make predictions transparent and interpretable.

Using SHAP, users can:

- Understand why a prediction was made
- Identify key demand drivers
- Analyze feature contributions
- Improve trust in model predictions

---

## 📋 Dashboard Components

### Forecast Overview

Provides:

- Daily Demand Prediction
- Forecast Horizon
- Total Forecast Demand
- Inventory Gap

### Inventory Risk Assessment

Provides:

- Inventory Sufficiency Check
- Shortage Detection
- Restocking Recommendations

### SHAP Explainability

Provides:

- Feature Importance Visualization
- Demand Driver Ranking
- Prediction Interpretation

### Business Analytics

Provides:

- Inventory vs Forecast Comparison
- Product and Store Profile

---

## 📸 Application Screenshots

### Forecast Dashboard

<img width="100%" alt="Forecast Dashboard" src="YOUR_SCREENSHOT_LINK_HERE">

### Inventory Risk Assessment

<img width="100%" alt="Inventory Assessment" src="YOUR_SCREENSHOT_LINK_HERE">

### SHAP Explainability

<img width="100%" alt="SHAP Analysis" src="YOUR_SCREENSHOT_LINK_HERE">

> Replace the image links above after uploading screenshots to the repository.

---

## 🎯 Key Highlights

- Built an end-to-end retail demand forecasting solution.
- Achieved an average R² score of **0.894** using 5-fold cross-validation.
- Improved forecasting performance using XGBoost.
- Integrated SHAP explainability for transparent predictions.
- Developed inventory shortage detection and risk assessment.
- Deployed as an interactive Streamlit web application.

---

## 🏗️ Project Structure

```text
DemandIQ/
│
├── app.py
├── demand_model.pkl
├── label_encoders1.pkl
├── feature_names.pkl
├── sales_data.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/dakshhkhatri/DemandIQ.git
cd DemandIQ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

- Multi-product forecasting
- Automated inventory replenishment recommendations
- Real-time demand prediction
- Deep learning-based forecasting models
- Cloud deployment and monitoring
- Advanced business intelligence dashboards

---

## 👨‍💻 Author

**Daksh Khatri**

GitHub:
https://github.com/dakshhkhatri

---

⭐ If you found this project interesting, consider giving it a star.
