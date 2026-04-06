# 1. Utilisation de Python 3.9 (plus stable pour tes modèles)
FROM python:3.9-slim

# 2. On définit le dossier de travail dans le conteneur
WORKDIR /app

# 3. Installation des outils de base
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. On copie le fichier des dépendances
COPY requirements.txt .

# 5. On installe les bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

# 6. On copie TOUT ton projet (Src, Models, Data, etc.)
COPY . .

# 7. On expose le port Streamlit
EXPOSE 8501

# 8. LA COMMANDE CRUCIALE (avec ton nom de fichier appmlops.py)
# On remplace CMD ["streamlit", "run", ...] par :
CMD ["python", "-m", "streamlit", "run", "Src/appmlops.py", "--server.port=8501", "--server.address=0.0.0.0"]