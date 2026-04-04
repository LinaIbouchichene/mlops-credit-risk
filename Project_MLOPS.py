import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# chemin correct du CSV
base_path = os.path.dirname(os.path.abspath(__file__))  # dossier MLops
file_path = os.path.join(base_path, 'Data', 'Loan_Data.csv')

# vérification du fichier
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Le fichier est introuvable ici : {file_path}")

# lecture du CSV
df = pd.read_csv(file_path)
print("📄 Aperçu du dataset :")
print(df.head())

def preprocess_data(filename):
    """Prétraitement des données : suppression ID, séparation X/y, train/test split, scaling."""
    # chemin du fichier
    file_path = os.path.join(base_path, 'Data', filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Le fichier est introuvable ici : {file_path}")

    # chargement du CSV
    df = pd.read_csv(file_path)

    # suppression de l'ID inutile
    df = df.drop('customer_id', axis=1)

    # séparation features et target
    X = df.drop('default', axis=1)
    y = df['default']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # sauvegarde du scaler
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
    joblib.dump(scaler, os.path.join(base_path, 'models', 'scaler.joblib'))

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, cols = preprocess_data("Loan_Data.csv")
        print("✅ Pré-traitement terminé avec succès !")
        print(f"📊 Nombre de variables : {len(cols)}")
        print(f"📈 Taille du jeu d'entraînement : {X_train.shape[0]} lignes")
    except Exception as e:
        print(e)

 

 
 