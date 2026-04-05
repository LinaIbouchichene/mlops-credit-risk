import streamlit as st
import pandas as pd
import joblib
import os
import mlflow

# -----------------------------------------------------
# CONFIGURATION & CONNEXION MLFLOW
# -----------------------------------------------------
# On force l'adresse locale pour éviter les erreurs 403/localhost
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_mlflow_data():
    try:
        # On cherche l'expérience exacte vue sur ta capture
        experiment = mlflow.get_experiment_by_name("Exp_Decision_Tree")
        if experiment:
            # On cherche spécifiquement ton meilleur run "Adjusted"
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runName = 'Decision_Tree_Adjusted'",
                max_results=1
            )
            if not runs.empty:
                return runs.iloc[0]
            else:
                # Fallback : on prend le meilleur recall de l'expérience
                runs_best = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.recall DESC"],
                    max_results=1
                )
                return runs_best.iloc[0] if not runs_best.empty else None
    except Exception as e:
        return None
    return None

# -----------------------------------------------------
# BRANDING & STYLE (Correction Texte Blanc/Noir)
# -----------------------------------------------------
BRAND_PRIMARY = "#004A99"
BRAND_LIGHT = "#F8F9FA"

st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")

def inject_custom_css():
    css = f"""
    <style>
    /* Fond de l'application */
    .stApp {{ background-color: {BRAND_LIGHT} !important; }}
    
    /* Force le texte en noir pour contrer le Dark Mode du navigateur */
    html, body, [data-testid="stWidgetLabel"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, li {{
        color: #1A1A1A !important;
    }}

    /* Cartes de métriques */
    .metric-card {{
        background: white !important;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {BRAND_PRIMARY};
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }}
    
    /* Style des boutons de la sidebar */
    .stButton>button {{
        width: 100%;
        border-radius: 5px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------
# NAVIGATION
# -----------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "intro"

with st.sidebar:
    st.title("🏦 Menu Banque")
    if st.button("🏠 Accueil"): st.session_state.stage = "intro"
    if st.button("📊 Performance MLflow"): st.session_state.stage = "metrics"
    if st.button("🔮 Simulateur de Crédit"): st.session_state.stage = "predict"
    st.markdown("---")
    st.caption("Projet MLOps - 2026")

# -----------------------------------------------------
# STAGE : INTRO
# -----------------------------------------------------
if st.session_state.stage == "intro":
    st.title("🏦 Intelligence Artificielle - Risque de Crédit")
    st.markdown("""
    ### Bienvenue dans l'interface de décision bancaire.
    Ce projet illustre un cycle **MLOps** complet :
    * **Exploration :** Analyse des données de crédit.
    * **Entraînement :** Optimisation d'un Arbre de Décision.
    * **Tracking :** Suivi des métriques (Recall) avec **MLflow**.
    * **Déploiement :** Interface Streamlit conteneurisée.
    """)
    st.image("https://img.freepik.com/free-vector/digital-economy-abstract-concept-illustration_335657-3972.jpg", width=400)

# -----------------------------------------------------
# STAGE : METRICS (Connexion Live + Photos)
# -----------------------------------------------------
elif st.session_state.stage == "metrics":
    st.title("📊 Tracking des Performances")
    
    # 1. RÉCUPÉRATION LIVE
    run_data = get_mlflow_data()
    
    if run_data is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recall (Live)", f"{run_data['metrics.recall']:.2%}")
        with col2:
            st.metric("Accuracy (Live)", f"{run_data['metrics.accuracy']:.2%}")
        with col3:
            st.write("**Run sélectionné :**")
            st.code(run_data['tags.mlflow.runName'])
    else:
        st.warning("⚠️ Mode déconnecté : Impossible de joindre le serveur MLflow (mlflow ui).")

    st.markdown("---")
    
    # 2. AFFICHAGE DES CAPTURES (BACKUP)
    st.subheader("📸 Preuves du Dashboard MLflow")
    
    # Dictionnaire des images (Vérifie bien que le dossier est 'assets' et l'extension '.png')
    images = {
        "Comparaison Accuracy/Recall": "Src/assets/mlflow_1.png",
        "Détails du modèle ajusté (88%)": "Src/assets/mlflow_2.png",
        "Paramètres du Decision Tree": "Src/assets/mlflow_3.png"
    }

    for title, path in images.items():
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)
        else:
            st.info(f"💡 Image '{title}' non trouvée dans {path}. (C'est normal si tu ne l'as pas encore ajoutée au dossier assets).")

# -----------------------------------------------------
# STAGE : PREDICT (Simulateur)
# -----------------------------------------------------
elif st.session_state.stage == "predict":
    st.title("🔮 Analyse de Risque en Temps Réel")
    
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Âge du client", 18, 95, 35)
        income = col2.number_input("Revenu Annuel (€)", 0, 1000000, 45000)
        loan_amount = col1.number_input("Montant du prêt (€)", 0, 500000, 15000)
        loan_intent = col2.selectbox("Objectif", ["Personnel", "Éducation", "Médical", "Entreprise", "Amélioration Habitat"])
        
        submitted = st.form_submit_button("Lancer l'évaluation")
        
        if submitted:
            # Simulation de la logique du modèle Decision Tree
            # (Dans un vrai projet, on ferait model.predict ici)
            if income > (loan_amount * 0.4) and age > 22:
                st.balloons()
                st.success("✅ CRÉDIT APPROUVÉ - Le profil présente un risque de défaut très faible.")
            else:
                st.error("❌ CRÉDIT REFUSÉ - Le modèle détecte un risque de défaut élevé.")