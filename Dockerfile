FROM python:3.11-slim

WORKDIR /app

# On installe les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask mlflow

# On copie le dossier Src
COPY Src/ ./Src/

# On expose le port Flask
EXPOSE 5000

# Commande de lancement (SANS le point à la fin !)
CMD ["python", "Src/appmlops.py"]