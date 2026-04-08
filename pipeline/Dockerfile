FROM python:3.10-slim

WORKDIR /app

# 1. Installation des dépendances
FROM python:3.10-slim

WORKDIR /app

# On s'assure que pip est à jour
RUN pip install --no-cache-dir --upgrade pip

# On installe explicitement streamlit + tes autres dépendances
# Si tu as d'autres bibliothèques (pandas, scikit-learn), ajoute-les ici ou vérifie requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le projet
COPY . .

# Exposition du port
EXPOSE 8501

# On utilise le chemin complet pour être sûr
ENTRYPOINT ["streamlit", "run", "pipeline/appml.py", "--server.port=8501", "--server.address=0.0.0.0"]

# 2. Copie de tout le projet (pour garder la structure /pipeline et /models)
COPY . .

# 3. Exposition du port par défaut de Streamlit
EXPOSE 8501

# 4. Commande de lancement CORRECTE
CMD ["streamlit", "run", "pipeline/appml.py", "--server.port=8501", "--server.address=0.0.0.0"]