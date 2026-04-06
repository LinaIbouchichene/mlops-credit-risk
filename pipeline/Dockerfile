FROM python:3.10-slim

WORKDIR /app

# 1. Copie et installe les dépendances (le fichier est maintenant à la racine)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copie le reste du code
COPY pipeline/ .
COPY models/ ./models/

# 3. Lance l'app (on est dans /app, donc appml.py est à la racine du conteneur)
CMD ["python", "appml.py"]