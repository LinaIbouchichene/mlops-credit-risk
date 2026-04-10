FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask mlflow

COPY Src/ ./Src/
COPY models/ ./models/
COPY mlruns/ ./mlruns/
COPY mlflow.db ./mlflow.db

EXPOSE 5000

CMD ["python", "Src/appmlops.py"]