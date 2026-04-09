
MLOPS PROJET
# mlops-credit-risk

# Branche develop1 : Zone d'Entraînement Itérative ( même process que la branche develop)

Cette branche est une extension de notre environnement de développement. Elle suit exactement le même processus rigoureux que la branche `develop` pour tester de nouvelles itérations du modèle.

## État d'avancement
Tous les processus MLOps ont été exécutés avec succès sur cette branche :
* **Prétraitement :** Nettoyage des données via `preprocess.py`.
* **Entraînement :** Exécution de `Model_ML.py` avec comparaison des scores.
* **Tracking :** Journalisation complète des métriques dans le dossier `mlruns`.
* **Sauvegarde :** Exportation des artefacts finaux dans le dossier `models/` via `Model_save.py`.
  
##  Structure de la Branche
La structure est identique à notre environnement de référence :
* **`pipeline/`** : Scripts d'automatisation des tâches.
* **`Src/`** : Code source de l'application en cours de test.
* **`Project_MLOPS.py`** : Script maître ayant piloté l'exécution globale.

## Workflow Exécuté
Comme pour la branche develop standard, le cycle suivant a été complété :
1.  Activation de l'environnement virtuel.
2.  Lancement du serveur **MLflow**.
3.  Exécution du pipeline complet pour valider la stabilité des nouveaux scripts.

---
