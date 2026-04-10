import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(filename):
    # Gestion du chemin automatique pour éviter l'erreur FileNotFoundError
    # On cherche le fichier dans le dossier 'Data' qui est au même niveau que 'Src'
    base_path = os.path.dirname(os.path.abspath(__file__)) # Dossier Src
    project_path = os.path.dirname(base_path) # Dossier MLOPS
    file_path = os.path.join(project_path, 'Data', filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Le fichier est introuvable ici : {file_path}")

    # 1. Chargement
    df = pd.read_csv(file_path)
    
    # 2. Nettoyage : Suppression de l'ID inutile
    df = df.drop('customer_id', axis=1)
    
    # 3. Séparation Features (X) et Target (y)
    X = df.drop('default', axis=1)
    y = df['default']
    
    # 4. Découpage Train/Test
    # On le fait ICI pour que le Scaler ne "voit" pas les données de test (Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Normalisation (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Sauvegarde du Scaler pour l'application future
    os.makedirs(os.path.join(project_path, 'models'), exist_ok=True)
    joblib.dump(scaler, os.path.join(project_path, 'models/scaler.joblib'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, cols = preprocess_data("Loan_Data.csv")
        print("✅ Pré-traitement terminé avec succès !")
        print(f"📊 Nombre de variables : {len(cols)}")
        print(f"📈 Taille du jeu d'entraînement : {X_train.shape[0]} lignes")
    except Exception as e:
        print(e)