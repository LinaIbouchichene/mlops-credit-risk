# mlops-credit-risk


#  Branche Develop : Zone d'Entraînement & Lab 🧪

Bienvenue sur la branche de Develop. Cette zone est dédiée à l'expérimentation, au tuning des modèles et à la validation des pipelines de données avant mise en production.

## Rôle de cette branche
Contrairement à la branche `main` qui héberge la version stable et déployée sur AWS, la branche `develop` sert à :
* **Expérimenter :** Tester de nouveaux algorithmes (XGBoost, SVM, etc.).
* **Tuning :** Ajuster les seuils de classification (Thresholds) pour optimiser le **Recall**.
* **Tracking :** Générer des runs MLflow intensifs pour comparer les performances.
* **Debug :** Corriger les scripts de prétraitement sans interrompre le service actif.

## État actuel des expérimentations
Actuellement, nous testons trois architectures :
1.  **Logistic Regression** : Baseline pour la rapidité.
2.  **Decision Tree** : Pour l'interprétabilité.
3.  **Random Forest** : Champion actuel en termes de robustesse.

## Workflow de travail sur cette zone
1.  **Lancer le serveur de tracking :** `mlflow server`
2.  **Exécuter l'entraînement :** `python mlops-credit-risk/Model_ML.py`
3.  **Analyser les résultats :** Comparer les métriques sur `localhost:5000`.
4.  **Sauvegarder les artefacts :** `python mlops-credit-risk/Model_save.py`

## Règles de la zone
* **Ne pas déployer directement :** Les modifications ici ne doivent être fusionnées avec `main` qu'après validation complète des métriques.
* **MLflow requis :** Tout entraînement doit être loggué pour garantir la reproductibilité.

---
🚀 *Une fois le modèle validé ici, effectuez une Pull Request vers `main` pour déclencher le déploiement automatique sur AWS.*

