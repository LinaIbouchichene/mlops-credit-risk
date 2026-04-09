
# mlops-credit-risk

#  Branche AWS : Infrastructure & Déploiement Cloud

Cette branche est dédiée à la configuration de l'infrastructure Cloud et à l'automatisation des pipelines de livraison continue (CD). Elle fait le lien entre les modèles entraînés et le déploiement sur AWS.

## Rôle de cette branche
Le but ici est d'assurer que chaque modification validée est envoyée en production de manière sécurisée et isolée via **Docker** et **AWS** :
* **CI/CD :** Automatisation du déploiement avec les workflows YAML.
* **Orchestration :** Gestion de l'image Docker contenant le pipeline complet.
* **Infrastructure :** Liaison avec Amazon ECR (stockage) et EC2 (exécution).

##  Structure de l'Infrastructure
* **`.github/workflows/`** : 
    * `aws.yml` : Workflow principal de déploiement vers AWS.
    * `github-docker-cicd.yaml` : Pipeline d'intégration et de build Docker.
* **`pipeline/`** : Scripts d'automatisation du flux de données.
* **`mlruns/` & `mlflow.db`** : Base de données de tracking locale pour validation finale avant push.
* **`models/`** : Artefacts (.pkl) prêts à être intégrés dans le conteneur.

## Composants de Production
* **`Project_MLOPS.py`** : Point d'entrée principal pour l'exécution sur le serveur.
* **`preprocess.py`** : Module de traitement des données en temps réel.
* **`Dockerfile`** : Recette de construction de l'environnement de production.

##  Processus de Déploiement
1. Les modèles sont validés techniquement dans la zone `pipeline`.
2. Le `Dockerfile` assemble le code source, les modèles et les dépendances.
3. Le workflow GitHub Actions pousse l'image vers **Amazon ECR**.
4. L'instance **Amazon EC2** récupère l'image et relance le service automatiquement.

---
**Note :** Les identifiants AWS ne sont pas stockés ici mais configurés dans les *GitHub Secrets* du dépôt.
