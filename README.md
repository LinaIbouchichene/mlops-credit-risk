# Projet MLOps - Credit Risk

Ce projet met en œuvre un pipeline MLOps complet pour l'évaluation du risque de crédit. L'objectif est d'automatiser le cycle de vie d'un modèle de Machine Learning, de l'entraînement au déploiement continu (CI/CD) sur AWS.

## 📋 Aperçu du Projet

L'application prédit la probabilité de défaut de paiement d'un client en fonction de son profil financier. Elle utilise des pratiques DevOps appliquées au ML pour garantir la reproductibilité et la stabilité du modèle.

## 🛠️ Stack Technique

* **Langage :** Python 3.x
* **Machine Learning :** Scikit-learn, Pandas, NumPy
* **Conteneurisation :** Docker
* **CI/CD :** GitHub Actions
* **Cloud (AWS) :**
    * **ECR :** Stockage des images Docker.
    * **EC2 :** Hébergement de l'application en production.
* **Versionnage :** Git

## 📂 Structure du Dossier .github/workflows

Le projet utilise GitHub Actions pour automatiser les tâches :
* `ci-cd.yml` : Gère les tests et l'intégration continue.
* `aws.yml` : Gère le déploiement automatique sur l'infrastructure AWS (ECR/EC2).

## 🚀 Installation Locale

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/LinaIbouchichene/mlops-credit-risk.git](https://github.com/LinaIbouchichene/mlops-credit-risk.git)
   cd mlops-credit-risk
