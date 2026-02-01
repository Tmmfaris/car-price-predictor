import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle, json

# -------- LOAD DATA --------
df = pd.read_csv("car_prediction_data (1).csv")

df["Car_Age"] = 2026 - df["Year"]
df.drop(["Car_Name","Year"], axis=1, inplace=True)

df = pd.get_dummies(df)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# save feature order
json.dump(list(X.columns), open("feature_columns.json","w"))

# -------- SPLIT --------
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# -------- SCALE --------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -------- MODELS (ONLY TWO) --------
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ),
    "LinearRegression": LinearRegression()
}

best_model = None
best_score = -1
best_name = ""

for name, m in models.items():
    m.fit(X_train_s, y_train)
    pred = m.predict(X_test_s)
    score = r2_score(y_test, pred)

    print(name, "R2:", score)

    if score > best_score:
        best_score = score
        best_model = m
        best_name = name

print("✅ Best model selected:", best_name)

# -------- SAVE --------
pickle.dump(best_model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
json.dump({"best_model": best_name}, open("model_info.json","w"))

print("✅ Best model saved")
