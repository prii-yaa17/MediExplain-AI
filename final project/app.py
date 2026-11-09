import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from utils.explainer import Explainer
from utils.pdf_generator import create_pdf_report
from utils.recommendations import HealthRecommender

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Medical Diagnosis AI", page_icon="🩺", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_models_and_metadata():
    models = {}
    models_dir = "models"
    
    # Check if models exist
    if not os.path.exists(models_dir) or not os.path.exists(os.path.join(models_dir, "scaler.pkl")):
        return None, None, None, None
        
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        with open(os.path.join(models_dir, f"{name}.pkl"), 'rb') as f:
            models[name.replace("_", " ").title()] = pickle.load(f)
            
    with open(os.path.join(models_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
        
    with open(os.path.join(models_dir, "features.pkl"), 'rb') as f:
        features = pickle.load(f)
        
    with open(os.path.join(models_dir, "model_metrics.pkl"), 'rb') as f:
        metrics = pickle.load(f)
        
    return models, scaler, features, metrics

models, scaler, feature_names, metrics = load_models_and_metadata()

if not models:
    st.error("Models not found! Please run `python utils/model_trainer.py` first.")
    st.stop()

# --- UI HEADER ---
st.title("🩺 Explainable AI-Based Medical Diagnosis System")
st.markdown("Predict heart disease risk and understand the reasoning using Multi-Model Machine Learning and SHAP.")

# --- SIDEBAR INPUT ---
st.sidebar.header("Patient Input Parameters")

# The requirements specifically asked for Age, BP, Cholesterol, Glucose, BMI
age = st.sidebar.slider("Age", 1, 100, 50)
sex_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
sex = 1 if sex_input == "Male" else 0

bmi = st.sidebar.number_input("BMI (kg/m²)", 10.0, 50.0, 25.0)

trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 500, 200)

fbs_input = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (High Glucose)", ["No", "Yes"])
fbs = 1 if fbs_input == "Yes" else 0

st.sidebar.markdown("---")
st.sidebar.subheader("Other Relevant Parameters")
cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
cp_inv = {v: k for k, v in cp_options.items()}
cp = cp_inv[st.sidebar.selectbox("Chest Pain Type", list(cp_options.values()))]

restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
restecg_inv = {v: k for k, v in restecg_options.items()}
restecg = restecg_inv[st.sidebar.selectbox("Resting ECG", list(restecg_options.values()))]

thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang_input = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang_input == "Yes" else 0

oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
slope_inv = {v: k for k, v in slope_options.items()}
slope = slope_inv[st.sidebar.selectbox("ST Slope", list(slope_options.values()))]

ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)

thal_options = {1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}
thal_inv = {v: k for k, v in thal_options.items()}
thal = thal_inv[st.sidebar.selectbox("Thalassemia", list(thal_options.values()))]


# Model Selection
st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox(
    "Select ML Model for Inference", 
    ["Auto (Best Model)", "Logistic Regression", "Random Forest", "Xgboost"]
)

# Compile features
patient_data_raw = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}

# Clone for health recommender logical access to prevent model schema mismatch
patient_data_recs = patient_data_raw.copy()
patient_data_recs['BMI'] = bmi

# Add BMI to our display dictionary (not used by model but shown in report)
patient_report_data = {
    'Age': age, 'Gender': sex_input, 'BMI': f"{bmi:.1f}", 'Blood Pressure': f"{trestbps} mmHg",
    'Cholesterol': f"{chol} mg/dl", 'High Glucose': fbs_input, 'Chest Pain': cp_options[cp]
}

df_input = pd.DataFrame([patient_data_raw])
df_scaled = scaler.transform(df_input)
df_scaled = pd.DataFrame(df_scaled, columns=feature_names)

# Resolve actual model
best_name = metrics.get('best_model', 'Random Forest')
if model_choice == "Auto (Best Model)":
    selected_model_name = best_name
else:
    selected_model_name = model_choice

selected_model = models[selected_model_name]

# --- INFERENCE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Prediction")
    
    # Run prediction
    prediction = selected_model.predict(df_scaled)[0]
    probabilities = selected_model.predict_proba(df_scaled)[0]
    
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    prob_score = probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100
    
    # Display Result
    if prediction == 1:
        st.error(f"### 🚨 {risk_label}")
        st.error(f"**Confidence**: {prob_score:.1f}%")
    else:
        st.success(f"### ✅ {risk_label}")
        st.success(f"**Confidence**: {prob_score:.1f}%")
        
    st.info(f"Using Model: **{selected_model_name}**")
        
    st.markdown("### Metrics Comparison")
    # Quick metric table
    metric_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Xgboost"],
        "Accuracy": [metrics["Logistic Regression"]["Accuracy"], metrics["Random Forest"]["Accuracy"], metrics["XGBoost"]["Accuracy"]],
        "F1 Score": [metrics["Logistic Regression"]["F1 Score"], metrics["Random Forest"]["F1 Score"], metrics["XGBoost"]["F1 Score"]]
    })
    st.dataframe(metric_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score'], color='lightgreen'))

with col2:
    st.subheader("🧠 Explainable AI (SHAP)")
    
    # SHAP Generation
    with st.spinner('Generating Explanations...'):
        explainer = Explainer(selected_model, feature_names)
        importance_df = explainer.get_local_explanation(df_scaled)
        
        # Natural Language
        nl_text = explainer.generate_natural_language_explanation(importance_df, prediction)
        st.markdown(f"**Diagnosis Summary:**\n*{nl_text}*")
        
        # Recommendations Generator
        st.markdown("### Personalized Health Recommendations")
        recommender = HealthRecommender()
        recs = recommender.generate_recommendations(patient_data_recs, importance_df, prediction)
        for r in recs:
            st.info(f"**💡 Tip:** {r['Tip']}  \n**🔍 Reason:** {r['Reason']}")
        
        # Bar Chart
        fig = explainer.plot_local_bar(importance_df)
        st.pyplot(fig)


# --- REPORT GENERATION ---
st.markdown("---")
st.subheader("📄 Generate Structured Report")
st.write("Download the complete diagnosis and explainability result as a PDF file.")

pdf_bytes = create_pdf_report(
    patient_data=patient_report_data,
    risk_label=risk_label,
    probability=prob_score,
    explanation_text=nl_text,
    plot_fig=fig,
    recommendations=recs
)

st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf_bytes,
    file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    mime="application/pdf",
    use_container_width=True
)

