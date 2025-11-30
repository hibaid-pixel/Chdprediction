import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------------
# SOLUTION À L'AttributeError: clean_categorical doit être défini 
# DANS app.py pour que joblib.load() puisse reconstruire le pipeline.
# -------------------------------------------------------------------------
def clean_categorical(df):
    """Uniformise les modalités de la variable famhist pour le pipeline."""
    # NOTE: Nous définissons cat_cols ici pour s'assurer que la fonction 
    # fonctionne sans dépendre de variables externes au pipeline.
    cat_cols = ['famhist'] 
    
    df = df.copy() 
    
    # Simuler le comportement d'un FunctionTransformer agissant sur l'ensemble du DataFrame
    for col in [c for c in df.columns if c in cat_cols]:
        # Application de strip().lower()
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.lower()
            
    return df
# -------------------------------------------------------------------------


# Configuration de la page
st.set_page_config(
    page_title="Prédiction du risque de CHD",
    page_icon="🫀",
    layout="centered"
)



# Créer deux colonnes principales (ratio 3:1 ou 2:1 pour un bon espace)
col_gauche, col_droite = st.columns([3, 1])



with col_gauche:
    st.title(" Application de prédiction du risque de maladie cardiaque (CHD)")
    st.write("""
    Cette application web a été **développée avec VS Code** et déployée avec **Streamlit**.
    Elle utilise un modèle de Machine Learning déjà entraîné et sauvegardé dans `Model.pkl`
    (pipeline : prétraitement + ACP + régression logistique) à partir du dataset *CHD.csv*.
    """)

    # 1. Chargement du modèle
    @st.cache_resource
    def load_model():
        try:
            model = joblib.load("Model.pkl")  
            return model
        except Exception as e:
            st.error(f"Erreur de chargement du modèle. Veuillez vérifier que 'Model.pkl' existe et que toutes les dépendances (y compris les classes scikit-learn) sont disponibles.")
            st.exception(e)
            return None

    model = load_model()

    # 2. Formulaire de saisie des variables
    st.subheader(" Saisir les informations du patient")

    if model:
        with st.form("chd_form"):
            form_col1, form_col2 = st.columns(2) # Colonnes pour organiser le formulaire
            
            # Note: J'ai utilisé un préfixe "form_" pour éviter les conflits de noms de colonnes
            with form_col1:
                age = st.number_input("Âge", min_value=15, max_value=70, value=45, help="Années")
                sbp = st.number_input("Pression systolique (sbp)", min_value=100.0, max_value=250.0, value=140.0, help="mmHg")
                ldl = st.number_input("LDL Cholestérol", min_value=10.0, max_value=1000.0, value=400.0, help="Concentration")
            
            with form_col2:
                adiposity = st.number_input("Adiposity", min_value=10.0, max_value=50.0, value=25.0, help="Mesure de graisse corporelle")
                obesity = st.number_input("Obesity", min_value=10.0, max_value=50.0, value=28.0, help="Indice d'obésité")
                famhist = st.selectbox("Antécédents familiaux (famhist)", ["Present", "Absent"])
            
            submitted = st.form_submit_button("Prédire le risque")


        # 3. Prédiction avec le modèle
        if submitted:
            input_data = {
                "sbp": sbp, "ldl": ldl, "adiposity": adiposity, 
                "famhist": famhist, "obesity": obesity, "age": age
            }
            
            input_df = pd.DataFrame([input_data])
            
            # 4. Affichage des résultats
            st.write("### Données saisies")
            st.dataframe(input_df)
            
            try:
                proba_chd = model.predict_proba(input_df)[0, 1]
                pred_chd = model.predict(input_df)[0]
                
                st.subheader(" Résultat de la prédiction")
                st.write(f"**Probabilité estimée de CHD (classe 1)** : `{proba_chd:.2f}`")
                
                if pred_chd == 1:
                    st.error(" Le modèle prédit **un risque élevé** de maladie cardiaque (CHD = 1).")
                else:
                    st.success(" Le modèle prédit **un risque faible** de maladie cardiaque (CHD = 0).")
                
                st.info(" Cette application est à but pédagogique et ne remplace pas un avis médical.")
                
            except Exception as e:
                st.error("Erreur lors de l'exécution de la prédiction.")
                st.exception(e)


