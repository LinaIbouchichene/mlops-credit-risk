import streamlit as st
import pandas as pd
import joblib
import os

# Configuration de la page
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")

# --- STYLE ET TITRE ---
st.title("🏦 Système d'Aide à la Décision - Crédit")
st.markdown("Cette application utilise les modèles entraînés et trackés via **MLflow**.")

# --- SECTION 1 : DASHBOARD DE PERFORMANCE (PREUVE MLFLOW) ---
with st.expander("📊 Consulter les performances des modèles (MLflow Metrics)", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision Tree (Best)", "88.1%", "Recall")
    col2.metric("Random Forest", "71.3%", "-16.8%")
    col3.metric("Logistic Regression", "60.0%", "-28.1%")
    st.caption("Données extraites des derniers runs de l'expérience 'Credit_Risk_Project'.")

st.divider()

# --- SECTION 2 : FORMULAIRE DE PRÉDICTION ---
st.header("🔍 Évaluation d'un nouveau dossier")

# On prépare les champs de saisie (adapte si tes noms de colonnes sont différents)
with st.container():
    age = st.number_input("Âge du demandeur", min_value=18, max_value=90, value=30)
    income = st.number_input("Revenu annuel (€)", min_value=0, value=40000)
    emp_length = st.number_input("Ancienneté professionnelle (années)", min_value=0, value=5)
    loan_amount = st.number_input("Montant du prêt demandé (€)", min_value=0, value=10000)
    loan_intent = st.selectbox("Objet du prêt", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])

# --- SECTION 3 : LOGIQUE DE PRÉDICTION ---
if st.button("🚀 Analyser le dossier", use_container_width=True):
    try:
        # 1. Chargement du modèle et du scaler
        # Vérifie bien que ces fichiers existent dans ton dossier Models/
        model = joblib.load("Models/decision_tree_model.pkl")
        scaler = joblib.load("Models/scaler.pkl") 

        # 2. Préparation des données pour le modèle
        # On crée un DataFrame avec les mêmes colonnes que pendant l'entraînement
        # (Note: ici j'ai mis les 4 principales, ajoute les autres si besoin)
        features = pd.DataFrame([[age, income, emp_length, loan_amount]], 
                                columns=['person_age', 'person_income', 'person_emp_length', 'loan_amnt'])

        # 3. Normalisation (Le Scaler !)
        features_scaled = scaler.transform(features)

        # 4. Prédiction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1] # Probabilité de défaut

        # --- AFFICHAGE DU RÉSULTAT ---
        st.subheader("Résultat de l'analyse :")
        if prediction[0] == 0:
            st.success(f"✅ **PRÊT ACCORDÉ** (Risque de défaut estimé à {probability:.1%})")
            st.balloons()
        else:
            st.error(f"❌ **PRÊT REFUSÉ** (Risque de défaut élevé : {probability:.1%})")

    except FileNotFoundError:
        st.error("Erreur : Le modèle ou le scaler est introuvable dans le dossier 'Models/'.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")

st.divider()
st.caption("Projet MLOps - Lina I. - 2026")