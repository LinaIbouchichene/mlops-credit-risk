import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(filename):
    # Chemin vers le fichier CSV
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'Data', filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Le fichier est introuvable ici : {file_path}")

    # 1. Chargement
    df = pd.read_csv(file_path)

    # 2. Nettoyage
    df = df.drop('customer_id', axis=1)
    
    # 3. Features / Target
    X = df.drop('default', axis=1)
    y = df['default']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Sauvegarde du scaler
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
    joblib.dump(scaler, os.path.join(base_path, 'models/scaler.joblib'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns