FROM python:3.10-slim

WORKDIR /app

# 1. On copie d'abord SEULEMENT le requirements.txt
COPY requirements.txt .

# 2. On installe les dépendances (Docker garde ça en cache)
FROM python:3.10-slim
WORKDIR /app
COPY . /app/

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "appml.py"]
