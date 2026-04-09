# Projet MLOps - Credit Risk

## Description du projet

Ce projet est un pipeline complet de **Machine Learning Operations (MLOps)** pour la prédiction du **risque de crédit** des clients.

Il permet de transformer un modèle ML classique en une solution industrialisée incluant :
- Préparation des données
- Entraînement de modèles
- Suivi des expériences (MLflow)
- Sauvegarde des modèles
- Automatisation CI/CD (GitHub Actions)
- Déploiement (Docker / AWS)

# Objectifs

- Analyser et préparer les données
- Construire plusieurs modèles de machine learning
- Comparer les performances des modèles
- Automatiser le pipeline ML
- Suivre les expériences avec MLflow
- Déployer le modèle en production

## Modèles utilisés

- Logistic Regression  
- Decision Tree  
- Random Forest  

  Objectif : comparer les performances et sélectionner le meilleur modèle.

##  Structure du projet

├── .github/workflows/ # CI/CD GitHub Actions
├── mlops-credit-risk/ # Code principal du projet
├── Model_ML.py # Entraînement des modèles
├── Model_save.py # Sauvegarde des modèles
├── Project_MLOPS.py # Pipeline principal
├── requirements.txt # Dépendances
└── README.md

---

##  Technologies utilisées

- Python  
- Scikit-learn  
- Pandas / NumPy 
- MLflow 
- Docker 
- GitHub Actions  
- AWS 
- Streamlit

- ## Pipeline MLOps

Le pipeline suit les étapes suivantes :  Data => Preprocessing => Training => Evaluation => Tracking =>  Deployment

## MLflow

MLflow est utilisé pour :
- suivre les expériences
- comparer les modèles
- enregistrer les métriques
- sauvegarder les modèles entraînés


## Docker

Le projet est conteneurisé avec Docker afin de garantir :
- reproductibilité
- portabilité
- facilité de déploiement


##  CI/CD (GitHub Actions)

Un workflow CI/CD est configuré pour :
- automatiser l’exécution du pipeline
- tester le code
- assurer l’intégration continue

## Déploiement

Le projet peut être déployé :
- en local
- avec Docker
- sur AWS


 
1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/LinaIbouchichene/mlops-credit-risk.git](https://github.com/LinaIbouchichene/mlops-credit-risk.git)
   cd mlops-credit-risk
