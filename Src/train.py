import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# --- FIX IMPORT ---
# On force Python à regarder dans le dossier actuel (Src)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocess import preprocess_data
    print("✅ Import de preprocess_data réussi !")
except ImportError as e:
    print(f"❌ Échec de l'import : {e}")
    sys.exit()

# 1. Préparation des données
X_train, X_test, y_train, y_test, cols = preprocess_data("Loan_Data.csv")

# 2. Modèles
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(max_depth=5),
    "Random_Forest": RandomForestClassifier(n_estimators=100)
}

# 3. Boucle MLflow (1 Modèle = 1 Expérience)
for name, model in models.items():
    mlflow.set_experiment(f"Exp_{name}")
    
    with mlflow.start_run(run_name="Initial_Run"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rec = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_params(model.get_params())
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"📊 {name} -> Recall: {rec:.4f}")

print("\n✨ C'est fini ! Tape 'mlflow ui' pour voir tes 3 expériences.")