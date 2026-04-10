import os
import joblib
import numpy as np
from preprocess import preprocess_data

# Charger les données
X_train, X_test, y_train, y_test, cols = preprocess_data("Loan_Data.csv")

# Chemin du dossier des modèles
models_path = os.path.join(os.getcwd(), "models")

# Définir modèles et thresholds
models_info = {
    "Logistic_Regression": {"file": "Logistic_Regression_model.joblib", "threshold": 0.6},
    "Decision_Tree": {"file": "Decision_Tree_model.joblib", "threshold": 0.7},
    "Random_Forest": {"file": "Random_Forest_model.joblib", "threshold": 0.75},
}

# Fonction de test rapide
def quick_test_model(y_pred, expected, name="Model"):
    try:
        if y_pred[0] == expected:
            print(f"{name}: Prediction correct ✅")
        else:
            print(f"{name}: Prediction incorrect ❌")
    except Exception:
        print(f"{name}: Prediction incorrect ❌")

# Boucle pour tester tous les modèles
for model_name, info in models_info.items():
    model_file = os.path.join(models_path, info["file"])
    
    if not os.path.exists(model_file):
        print(f"❌ Le modèle {model_name} n'existe pas : {model_file}")
        continue
    
    # Charger le modèle
    model = joblib.load(model_file)
    
    # Calcul des prédictions
    if hasattr(model, "predict_proba"):
        threshold = info["threshold"]
        y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    # Tester
    quick_test_model(y_pred, expected=y_test.iloc[0], name=model_name)