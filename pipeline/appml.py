# appml.py
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

# Charger les modèles ML sauvegardés localement
models = {
    "Decision Tree": joblib.load(os.path.join(models_path, "Decision_Tree_model.joblib")),
    "Random Forest": joblib.load(os.path.join(models_path, "Random_Forest_model.joblib")),
    "Logistic Regression": joblib.load(os.path.join(models_path, "Logistic_Regression_model.joblib"))
}

# Scores de chaque modèle (à adapter selon ta validation)
model_scores = {
    "Decision Tree": 0.82,
    "Random Forest": 0.88,
    "Logistic Regression": 0.80
}

# Sélection automatique du meilleur modèle
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

# Fonction de prédiction générique (aucune info sur les features n'est affichée)
def predict_best_model():
    # Ici tu peux créer un DataFrame factice ou prendre un exemple prétraité
    # Pour l'exemple, on utilise un vecteur vide ou constant
    import numpy as np
    dummy_input = pd.DataFrame([np.zeros(len(scaler.mean_))])
    dummy_scaled = scaler.transform(dummy_input)
    pred = best_model.predict(dummy_scaled)
    return int(pred[0])

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    prediction = predict_best_model()
    
    # Message simple et opaque
    if prediction == 1:
        message = f"⚠️ {best_model_name} prédit un risque de défaut !"
    else:
        message = f"✅ {best_model_name} prédit que le client est fiable."

    return render_template("index.html", prediction_text=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)