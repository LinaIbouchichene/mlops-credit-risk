# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --- Charger les modèles et le scaler ---
base_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(base_path, "models")

# Charger le scaler
scaler = joblib.load(os.path.join(models_path, "scaler.joblib"))

# Charger les modèles MLflow sauvegardés localement
dt_model = joblib.load(os.path.join(models_path, "Decision_Tree_model.joblib"))
rf_model = joblib.load(os.path.join(models_path, "Random_Forest_model.joblib"))
log_model = joblib.load(os.path.join(models_path, "Logistic_Regression_model.joblib"))

# Fonction de prédiction rapide
def model_pred(model, features):
    df = pd.DataFrame([features])
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    return int(pred[0])

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Récupération des inputs depuis le formulaire
        Age = float(request.form["Age"])
        RestingBP = float(request.form["RestingBP"])
        Cholesterol = float(request.form["Cholesterol"])
        Oldpeak = float(request.form["Oldpeak"])
        FastingBS = float(request.form["FastingBS"])
        MaxHR = float(request.form["MaxHR"])

        features = {
            "Age": Age,
            "RestingBP": RestingBP,
            "Cholesterol": Cholesterol,
            "FastingBS": FastingBS,
            "MaxHR": MaxHR,
            "Oldpeak": Oldpeak
        }

        # Choisir le modèle ici (exemple: Decision Tree)
        prediction = model_pred(dt_model, features)

        if prediction == 1:
            message = "⚠️ Le client présente un risque de défaut !"
        else:
            message = "✅ Le client semble fiable."

        return render_template("index.html", prediction_text=message)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)