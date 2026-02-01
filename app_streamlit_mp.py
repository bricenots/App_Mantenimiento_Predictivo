import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Configuración general
# ===============================
st.set_page_config(
    page_title="Mantenimiento Predictivo",
    layout="centered"
)

# ===============================
# Carga de artefactos
# ===============================
model = joblib.load("artifacts/model_clf_solemne1.pkl")

with open("artifacts/metrics_solemne1.json", "r") as f:
    metrics = json.load(f)

# Dataset solo para visualización
df = pd.read_csv("ai4i2020.csv")

# Variables usadas
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# ===============================
# Título y contexto
# ===============================
st.title("Mantenimiento Predictivo – Clasificación de Fallas")

st.markdown("""
Esta aplicación permite **evaluar el riesgo de falla de maquinaria**
utilizando un modelo de **clasificación supervisada**, entrenado y evaluado
en la **Solemne 1** del curso.

El usuario puede **simular un caso nuevo** y visualizar su posición
respecto al comportamiento histórico del dataset.
""")

st.divider()

# ===============================
# Resultados del modelo
# ===============================
st.header("1️⃣ Resultados del Modelo")

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", round(metrics["accuracy"], 3))
with col2:
    st.metric("Recall (Falla)", round(metrics["recall_failure"], 3))

st.subheader("Matriz de Confusión")

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(
    metrics["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No Falla", "Falla"],
    yticklabels=["No Falla", "Falla"],
    ax=ax
)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)

st.markdown("""
**Interpretación operacional**

- El modelo presenta una **alta exactitud global**, influenciada por el desbalance del dataset.
- El **recall de la clase Falla (~63%)** indica una capacidad razonable de detección de fallas reales.
- Los **falsos negativos** representan el principal riesgo operacional.
""")

st.divider()

# ===============================
# Simulación de caso nuevo
# ===============================
# Rangos reales del dataset
min_vals = df[features].min()
max_vals = df[features].max()

st.header("2️⃣ Simulación de un Caso Nuevo")
st.markdown(
    f"""
**Rangos del dataset**  
- Tool wear: {min_vals['Tool wear [min]']} – {max_vals['Tool wear [min]']}  
- Torque: {round(min_vals['Torque [Nm]'],1)} – {round(max_vals['Torque [Nm]'],1)}  
- Rotational speed: {min_vals['Rotational speed [rpm]']} – {max_vals['Rotational speed [rpm]']}
"""
)

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air temperature [K]",
        float(min_vals["Air temperature [K]"]),
        float(max_vals["Air temperature [K]"]),
        float(df["Air temperature [K]"].mean())
    )
    proc_temp = st.number_input(
        "Process temperature [K]",
        float(min_vals["Process temperature [K]"]),
        float(max_vals["Process temperature [K]"]),
        float(df["Process temperature [K]"].mean())
    )
    rot_speed = st.number_input(
        "Rotational speed [rpm]",
        int(min_vals["Rotational speed [rpm]"]),
        int(max_vals["Rotational speed [rpm]"]),
        int(df["Rotational speed [rpm]"].mean())
    )

with col2:
    torque = st.number_input(
        "Torque [Nm]",
        float(min_vals["Torque [Nm]"]),
        float(max_vals["Torque [Nm]"]),
        float(df["Torque [Nm]"].mean())
    )
    tool_wear = st.number_input(
        "Tool wear [min]",
        int(min_vals["Tool wear [min]"]),
        int(max_vals["Tool wear [min]"]),
        int(df["Tool wear [min]"].mean())
    )

# ===============================
# Predicción
# ===============================
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

    if pred == "FALLA":
        st.error(f"⚠️ Riesgo de FALLA\n\nProbabilidad: {round(proba,3)}")
    else:
        st.success(f"✅ Sin falla esperada\n\nProbabilidad: {round(proba,3)}")

    # ===============================
    # Visualización de posición del usuario
    # ===============================
    st.subheader("📍 Posición del caso evaluado en el dataset")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Tool wear vs Torque
    sns.scatterplot(
        data=df,
        x="Tool wear [min]",
        y="Torque [Nm]",
        hue="Machine failure",
        alpha=0.3,
        ax=axs[0],
        legend=False
    )
    axs[0].scatter(
        tool_wear, torque,
        color="red", s=120, edgecolor="black", label="Caso evaluado"
    )
    axs[0].set_title("Tool wear vs Torque")

    # Rotational speed vs Torque
    sns.scatterplot(
        data=df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue="Machine failure",
        alpha=0.3,
        ax=axs[1],
        legend=False
    )
    axs[1].scatter(
        rot_speed, torque,
        color="red", s=120, edgecolor="black", label="Caso evaluado"
    )
    axs[1].set_title("Rotational speed vs Torque")

    st.pyplot(fig)

# ===============================
# Cierre académico
# ===============================
st.divider()
st.caption("""
Modelo entrenado y evaluado en la Solemne 1.  
Resultados reproducidos sin ajustes posteriores.  
Aplicación desarrollada en Streamlit.
""")
