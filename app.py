from flask import Flask, render_template, request
import numpy as np
import pickle, json

app = Flask(__name__)

# -------- LOAD MODEL FILES --------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
FEATURE_COLUMNS = json.load(open("feature_columns.json"))

try:
    MODEL_NAME = json.load(open("model_info.json"))["best_model"]
except:
    MODEL_NAME = "AutoModel"

CURRENT_YEAR = 2026


# -------- HOME --------
@app.route("/")
def home():
    return render_template("index.html")


# -------- PREDICT --------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ---- read form ----
        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        owner = int(request.form["owner"])

        fuel = request.form["fuel_type"]
        seller = request.form["seller_type"]
        trans = request.form["transmission"]

        # ---- feature engineering ----
        car_age = CURRENT_YEAR - year

        row = {col: 0 for col in FEATURE_COLUMNS}

        def safe_set(k, v):
            if k in row:
                row[k] = v

        safe_set("Present_Price", present_price)
        safe_set("Kms_Driven", kms_driven)
        safe_set("Owner", owner)
        safe_set("Car_Age", car_age)

        safe_set(f"Fuel_Type_{fuel}", 1)
        safe_set(f"Seller_Type_{seller}", 1)
        safe_set(f"Transmission_{trans}", 1)

        # ---- align feature order ----
        X = np.array([[row[c] for c in FEATURE_COLUMNS]])
        Xs = scaler.transform(X)

        # ---- prediction ----
        pred = model.predict(Xs)[0]
        prediction = round(pred, 2)

        # ---- confidence score ----
        if hasattr(model, "estimators_"):
            tree_preds = [t.predict(Xs)[0] for t in model.estimators_]
            confidence = max(0, min(100, int(100 - np.std(tree_preds) * 10)))
        else:
            confidence = 80

        # ---- market range band ----
        low = round(prediction * 0.9, 2)
        high = round(prediction * 1.1, 2)

        # ---- render result ----
        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            model_name=MODEL_NAME,
            low=low,
            high=high
        )

    except Exception as e:
        print("ERROR:", e)

        return render_template(
            "result.html",
            prediction="Error",
            confidence=0,
            model_name=MODEL_NAME,
            low=0,
            high=0
        )


# -------- RUN --------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

