from flask import Flask, render_template_string, request
import mlflow
import joblib
import numpy as np

app = Flask(__name__)

import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE}/mlflow.db")
MODEL_PATH = os.getenv("MODEL_PATH", f"{BASE}/mlruns/3/models/m-9a88b116f8274045964b95b50cb04777/artifacts/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", f"{BASE}/models/scaler.joblib")

# Chargement du modèle et du scaler au démarrage
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Erreur chargement modèle: {e}")

def get_metrics():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("Exp_Decision_Tree")
        runs = client.search_runs(exp.experiment_id, max_results=1)
        if runs:
            m = runs[0].data.metrics
            return {
                "recall": round(m.get("recall", 0) * 100, 1),
                "accuracy": round(m.get("accuracy", 0) * 100, 1)
            }
    except:
        pass
    return {"recall": "N/A", "accuracy": "N/A"}

def predict(credit_lines, loan_amt, total_debt, income, years_employed, fico_score):
    features = np.array([[credit_lines, loan_amt, total_debt, income, years_employed, fico_score]])
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, round(proba * 100, 1)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Credit Risk Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background-color: #f8f9fa; }
        .metric-val { font-size: 2rem; font-weight: 700; }
        .card { border: none; border-radius: 12px; }
        .badge-model { background: #0d6efd22; color: #0d6efd; font-size: 0.75rem; padding: 4px 10px; border-radius: 20px; }
        .result-box { border-radius: 12px; font-size: 1.1rem; font-weight: 600; }
    </style>
</head>
<body class="container py-5" style="max-width: 860px;">

    <div class="d-flex align-items-center justify-content-between mb-4">
        <div>
            <h2 class="mb-0 fw-bold">Credit Risk Dashboard</h2>
            <span class="badge-model mt-1 d-inline-block">Decision Tree — Meilleur modèle</span>
        </div>
    </div>

    <!-- Métriques -->
    <div class="row g-3 mb-4">
        <div class="col-6">
            <div class="card p-4 shadow-sm text-center">
                <div class="text-muted small mb-1">Accuracy</div>
                <div class="metric-val text-primary">{{ metrics.accuracy }}%</div>
                <div class="progress mt-2" style="height:6px;">
                    <div class="progress-bar bg-primary" style="width: {{ metrics.accuracy }}%"></div>
                </div>
            </div>
        </div>
        <div class="col-6">
            <div class="card p-4 shadow-sm text-center">
                <div class="text-muted small mb-1">Recall</div>
                <div class="metric-val text-success">{{ metrics.recall }}%</div>
                <div class="progress mt-2" style="height:6px;">
                    <div class="progress-bar bg-success" style="width: {{ metrics.recall }}%"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Simulateur -->
    <div class="card p-4 shadow-sm">
        <h5 class="fw-semibold mb-3">Simulateur de risque de crédit</h5>
        <form method="POST">
            <div class="row g-3">
                <div class="col-md-6">
                    <label class="form-label text-muted small">Revenu annuel ($)</label>
                    <input type="number" name="income" class="form-control" placeholder="ex: 50000" value="{{ form.income }}" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label text-muted small">Montant du prêt ($)</label>
                    <input type="number" name="loan_amt" class="form-control" placeholder="ex: 10000" value="{{ form.loan_amt }}" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label text-muted small">Dette totale ($)</label>
                    <input type="number" name="total_debt" class="form-control" placeholder="ex: 5000" value="{{ form.total_debt }}" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label text-muted small">Lignes de crédit actives</label>
                    <input type="number" name="credit_lines" class="form-control" placeholder="ex: 3" value="{{ form.credit_lines }}" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label text-muted small">Années d'emploi</label>
                    <input type="number" name="years_employed" class="form-control" placeholder="ex: 5" value="{{ form.years_employed }}" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label text-muted small">Score FICO (300–850)</label>
                    <input type="number" name="fico_score" class="form-control" placeholder="ex: 680" min="300" max="850" value="{{ form.fico_score }}" required>
                </div>
            </div>
            <button type="submit" class="btn btn-primary mt-3 px-4">Analyser le risque</button>
        </form>

        {% if result is not none %}
        <div class="mt-4 p-3 result-box {{ 'bg-success bg-opacity-10 text-success' if result == 0 else 'bg-danger bg-opacity-10 text-danger' }}">
            {% if result == 0 %}
                Credit APPROUVE — Probabilite de defaut : {{ proba }}%
            {% else %}
                Credit REFUSE — Probabilite de defaut : {{ proba }}%
            {% endif %}
            <div class="progress mt-2" style="height: 8px;">
                <div class="progress-bar {{ 'bg-success' if result == 0 else 'bg-danger' }}" style="width: {{ proba }}%"></div>
            </div>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    metrics = get_metrics()
    result = None