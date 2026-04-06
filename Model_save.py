from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Exemple de données
X, y = make_classification(n_samples=100, n_features=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation si besoin
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement des modèles
dt_model = DecisionTreeClassifier(max_depth=5, class_weight={0:1, 1:0.02})
dt_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=30, max_depth=4, class_weight={0:1, 1:0.005}, random_state=42)
rf_model.fit(X_train_scaled, y_train)

log_model = LogisticRegression(max_iter=1000, class_weight={0:1, 1:0.03})
log_model.fit(X_train_scaled, y_train)

# Créer le dossier models s'il n'existe pas
models_path = os.path.join(os.getcwd(), "models")
os.makedirs(models_path, exist_ok=True)

# Sauvegarder le scaler et les modèles
joblib.dump(scaler, os.path.join(models_path, "scaler.joblib"))
joblib.dump(dt_model, os.path.join(models_path, "Decision_Tree_model.joblib"))
joblib.dump(rf_model, os.path.join(models_path, "Random_Forest_model.joblib"))
joblib.dump(log_model, os.path.join(models_path, "Logistic_Regression_model.joblib"))

print("✅ Modèles et scaler sauvegardés dans le dossier models/")