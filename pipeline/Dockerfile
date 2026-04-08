FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le projet
COPY . .

# Flask utilise le port 5001 d'après ton code
EXPOSE 5001

# COMMANDE CRUCIALE : On lance python, PAS streamlit
CMD ["python", "pipeline/appml.py"]