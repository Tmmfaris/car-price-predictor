# 🚗 Car Price Prediction Web Application

A machine learning powered web application that predicts car prices based on user inputs. The system is built using Python and Flask, integrates trained ML models, and is deployed on the cloud for real-time predictions.

This project demonstrates practical skills in **Machine Learning, Model Deployment, Flask Web Development, and Cloud Hosting**.

---

## 🌐 Live Demo

🔗 **Live App:** https://web-production-83b8a.up.railway.app

---

## 🎥 Demo Video

[🔗 Watch Demo Video](https://drive.google.com/file/d/1q54QTeHYb-OcXXEgfDooCNoxhnqRS2qx/view?usp=sharing)

---

## 🧠 Problem Statement

Car price estimation is important for buyers and sellers. This project builds a regression-based ML system that predicts fair car prices using historical vehicle data and feature engineering.

---

## ⚙️ Tech Stack

### Languages & Frameworks
- Python
- Flask

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Models Used
- Random Forest Regressor
- Linear Regression

### Frontend
- HTML
- CSS
- Bootstrap

### Deployment
- Railway Cloud Platform

---

## ✨ Features

- Real-time car price prediction
- Multiple ML model comparison
- Preprocessing + scaling pipeline
- Clean web UI for user inputs
- Serialized trained model usage
- Cloud deployed application

---

## 📂 Project Structure

```
car-price-predictor/
│
├── app.py
├── train_model.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
├── model.pkl
├── scaler.pkl
├── feature_columns.json
├── requirements.txt
└── README.md
```

---


### ▶️ Run Flask App

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 ML Workflow

- Data cleaning
- Feature engineering
- Model training
- Model evaluation (R² comparison)
- Best model selection
- Model serialization (pickle)
- Flask integration

---

## 📌 Prediction Input Features

The model generates price estimates based on the following vehicle attributes:

- Model Year
- Showroom Price (Lakhs)
- Kilometers Driven
- Owner Count
- Fuel Type (Petrol / Diesel / CNG)
- Seller Type (Dealer / Individual)
- Transmission (Manual / Automatic)


## 🚀 Deployment

Deployed on Railway with GitHub integration for automatic redeployment.

---

## 🔮 Future Improvements

- Larger dataset training
- Deep learning regression model
- Prediction confidence interval
- User analytics dashboard

---