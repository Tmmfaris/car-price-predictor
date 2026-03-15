# Car Price Prediction Engine

A **Machine Learning powered web application** that predicts the resale value of cars based on vehicle attributes and market factors.  
The system uses trained regression models and a Flask web interface to deliver **real-time predictions through an interactive UI**.

This project demonstrates practical skills in:

- Machine Learning
- Data Preprocessing & Feature Engineering
- Model Deployment
- Flask Web Development
- Cloud Hosting

---

# Live Demo

🔗 **Live Application**  
https://car-price-predictor-8d8e.onrender.com

---

# Demo Video

🔗 **Watch Demo Video**  
https://drive.google.com/file/d/1q54QTeHYb-OcXXEgfDooCNoxhnqRS2qx/view?usp=sharing

---

# Google Colab Notebook

The complete **machine learning training workflow** for this project is available in Google Colab.

🔗 **Open in Colab**  
https://colab.research.google.com/drive/1cmf7R4E_o92ekEqGrpJNrPvnmN2uiqeL?usp=sharing

The notebook includes:

- Dataset preprocessing
- Feature engineering
- Model training
- Model comparison
- Model evaluation
- Exporting trained model for deployment

---

# Problem Statement

Car price estimation is important for both buyers and sellers in the used car market.  
This project builds a **machine learning regression system** that predicts fair resale prices using historical vehicle data and engineered features.

The goal is to provide **accurate, data-driven price estimation** through an easy-to-use web interface.

---

# Machine Learning Approach

The system uses supervised learning regression models trained on vehicle attributes.

### ML Workflow

1. Data Cleaning  
2. Feature Engineering  
3. Model Training  
4. Model Evaluation  
5. Best Model Selection  
6. Model Serialization  
7. Flask Integration  
8. Web Deployment  

---

# Tech Stack

## Programming & Framework
- Python
- Flask

## Data Processing
- Pandas
- NumPy

## Machine Learning
- Scikit-learn

## Models Used
- Random Forest Regressor
- Linear Regression

## Frontend
- HTML
- CSS
- Bootstrap

## Deployment
- Render Cloud Platform

---

# Features

- AI-powered car price estimation
- Real-time resale value prediction
- Interactive web interface
- Feature engineering pipeline
- Multiple ML model comparison
- Serialized trained model integration
- Cloud-hosted Flask application

---

# Prediction Input Features

The model predicts resale value using the following vehicle attributes:

- Model Year
- Showroom Price (Lakhs)
- Kilometers Driven
- Owner Count
- Fuel Type (Petrol / Diesel / CNG)
- Seller Type (Dealer / Individual)
- Transmission Type (Manual / Automatic)

---

# Project Structure

```
car-price-predictor/
│
├── app.py
├── train_model.py
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│
├── model.pkl
├── scaler.pkl
├── feature_columns.json
│
├── requirements.txt
└── README.md
```

---

# Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Tmmfaris/car-price-predictor.git
```

```
cd car-price-predictor
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

# Deployment

The application is deployed using **Render Cloud Platform** with GitHub integration.

Features:

- Automatic deployment
- Continuous integration
- Real-time model prediction

---

# Future Improvements

- Larger dataset training
- Deep learning regression models
- Prediction confidence intervals
- Car market analytics dashboard
- Model explainability (SHAP / LIME)

---

# Author

**Muhammed Faris T M**

📧 Email: tmmfaris@gmail.com  
🔗 GitHub: https://github.com/Tmmfaris  
🔗 LinkedIn: http://www.linkedin.com/in/muhammed-faris-tm-ab1233196

---
