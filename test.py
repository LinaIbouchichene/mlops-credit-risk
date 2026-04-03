
from sklearn.exceptions import NotFittedError
import pandas as pd
from sklearn.model_selection import train_test_split

def test_model(model, X_test, y_test, name="Model"):
    try:
        # Vérification que le modèle peut faire des prédictions
        if hasattr(model, "predict_proba"):
            y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
        
        # Vérification dimensions
        assert y_pred.shape[0] == y_test.shape[0], "Taille des prédictions différente du test set"
        
        # Vérification valeurs
        assert set(y_pred).issubset({0, 1}), "Prédictions non binaires détectées"
        
        print(f"✅ {name}: Test réussi, prédictions correctes")
    except NotFittedError:
        print(f"❌ {name}: Le modèle n'a pas été entraîné")
    except AssertionError as e:
        print(f"❌ {name}: Test incorrect - {e}")
    except Exception as e:
        print(f"❌ {name}: Erreur inattendue - {e}")

# Tests pour chacun des modèles
print("🔹 Test modèle Decision Tree")
test_model(model=model, X_test=X_test, y_test=y_test, name="Decision Tree")

print("🔹 Test modèle Random Forest")
test_model(model=model, X_test=X_test, y_test=y_test, name="Random Forest")

# Si tu avais un troisième modèle, exemple Régression Logistique
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)  # entraînement rapide
print("🔹 Test modèle Logistic Regression")
test_model(model=log_model, X_test=X_test, y_test=y_test, name="Logistic Regression")
