import streamlit as st
import pandas as pd
import joblib
import json

# =========================================
# Configuración
# =========================================
st.set_page_config(page_title="Mantenimiento Predictivo", layout="centered")

# =========================================
# Carga de artefactos
# =========================================
model = joblib.load("artifacts/model_clf_solemne1.pkl")

with open("artifacts/metrics_solemne1.json", "r") as f:
    metrics = json.load(f)

# =========================================
# Título
# =========================================
st.title("Aplicación de Mantenimiento Predictivo")
st.write("Clasificador entrenado con datos de Solemne 1")

# =========================================
# Resultados del modelo
# =========================================
st.header("Resultados del Modelo")

st.metric("Accuracy", round(metrics["accuracy"], 3))
st.metric("Recall (Falla)", round(metrics["recall_failure"], 3))

st.subheader("Matriz de Confusión")
cm_df = pd.DataFrame(
    metrics["confusion_matrix"],
    columns=["Pred No Falla", "Pred Falla"],
    index=["Real No Falla", "Real Falla"]
)
st.dataframe(cm_df)

# =========================================
# Prueba del modelo
# =========================================
st.header("Prueba del Modelo")

air_temp = st.number_input("Air temperature [K]", 290.0, 320.0, 300.0)
proc_temp = st.number_input("Process temperature [K]", 300.0, 350.0, 310.0)
rot_speed = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
tool_wear = st.number_input("Tool wear [min]", 0, 300, 50)

if st.button("Clasificar"):
    input_data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    proba = model.predict_proba(pd.DataFrame([input_data]))[0][1]
    pred = "FALLA" if proba >= 0.5 else "NO FALLA"

    st.subheader("Resultado")
    st.write(f"Clasificación: **{pred}**")
    st.write(f"Probabilidad de falla: **{round(proba, 3)}**")
