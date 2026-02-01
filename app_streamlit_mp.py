import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np

# ======================================
# Configuración
# ======================================
st.set_page_config(
    page_title="Simulador Operativo – Mantenimiento Predictivo",
    layout="centered"
)

MACHINE_LABEL = "Máquina industrial genérica – Dataset AI4I 2020"

# ======================================
# Carga de artefactos
# ======================================
model = joblib.load("artifacts/model_clf_solemne1.pkl")

with open("artifacts/metrics_solemne1.json", "r") as f:
    metrics = json.load(f)

df = pd.read_excel("artifacts/dataset_final_solemne1.xlsx")

features = [
    "Torque [Nm]",
    "Tool wear [min]",
    "Rotational speed [rpm]",
    "Process temperature [K]",
    "Air temperature [K]"
]

mins = df[features].min()
maxs = df[features].max()
means = df[features].mean()

pct = df[features].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).T

# ======================================
# Funciones operativas
# ======================================
def semaphore_status(var, value):
    p10, p25, p50, p75, p90 = pct.loc[var, [0.10, 0.25, 0.50, 0.75, 0.90]]

    if value < p10 or value > p90:
        return "CRÍTICO", "🔴", "#ff4d4d"
    elif value < p25 or value > p75:
        return "ALERTA", "🟠", "#ffb347"
    else:
        return "NORMAL", "🟢", "#4CAF50"

def draw_gauge(var, value, unit):
    status, icon, color = semaphore_status(var, value)
    fig, ax = plt.subplots(figsize=(6, 0.8))

    ax.barh([0], [value], color=color)
    ax.axvline(means[var], color="black", linestyle="--", linewidth=1)

    ax.set_xlim(mins[var], maxs[var])
    ax.set_yticks([])
    ax.set_title(f"{icon} {var} → {status}", loc="left", fontsize=11)
    ax.set_xlabel(unit)
    plt.tight_layout()
    return fig, status

# ======================================
# CABECERA
# ======================================
st.title("🛠️ Simulador Operativo de Mantenimiento Predictivo")
st.caption(MACHINE_LABEL)

st.info(
    "Vista operativa. El semáforo indica **qué variable está más comprometida**. "
    "Ajusta primero las que estén en rojo."
)

# ======================================
# VALORES INICIALES (promedios)
# ======================================
torque = float(means["Torque [Nm]"])
tool_wear = int(means["Tool wear [min]"])
rot_speed = int(means["Rotational speed [rpm]"])
proc_c = float(means["Process temperature [K]"] - 273.15)
air_c = float(means["Air temperature [K]"] - 273.15)

input_data = {
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Rotational speed [rpm]": rot_speed,
    "Process temperature [K]": proc_c + 273.15,
    "Air temperature [K]": air_c + 273.15
}

# ======================================
# PANEL SUPERIOR – SEMÁFOROS
# ======================================
st.header("🚦 Estado actual de la máquina")

for var, unit in [
    ("Torque [Nm]", "Nm"),
    ("Tool wear [min]", "min"),
    ("Rotational speed [rpm]", "rpm"),
    ("Process temperature [K]", "°C"),
    ("Air temperature [K]", "°C"),
]:
    val = input_data[var]
    fig, status = draw_gauge(var, val, unit)
    st.pyplot(fig)
    if "temperature" in var.lower():
        st.caption("Variable contextual – no prioritaria")

st.divider()

# ======================================
# SIMULADOR
# ======================================
st.header("🔧 Ajuste de parámetros (simulación)")

torque = st.slider(
    "Torque [Nm] (PRIORIDAD)",
    float(mins["Torque [Nm]"]),
    float(maxs["Torque [Nm]"]),
    torque
)

tool_wear = st.slider(
    "Tool wear [min]",
    int(mins["Tool wear [min]"]),
    int(maxs["Tool wear [min]"]),
    tool_wear
)

rot_speed = st.slider(
    "Rotational speed [rpm]",
    int(mins["Rotational speed [rpm]"]),
    int(maxs["Rotational speed [rpm]"]),
    rot_speed
)

st.markdown("### 🌡️ Variables contextuales")

air_c = st.slider(
    "Air temperature [°C]",
    float(mins["Air temperature [K]"] - 273.15),
    float(maxs["Air temperature [K]"] - 273.15),
    air_c
)

proc_c = st.slider(
    "Process temperature [°C]",
    float(mins["Process temperature [K]"] - 273.15),
    float(maxs["Process temperature [K]"] - 273.15),
    proc_c
)

input_data = {
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Rotational speed [rpm]": rot_speed,
    "Process temperature [K]": proc_c + 273.15,
    "Air temperature [K]": air_c + 273.15
}

# ======================================
# RESULTADO DEL MODELO
# ======================================
st.divider()
st.header("📊 Evaluación de riesgo")

proba = model.predict_proba(pd.DataFrame([input_data]))[0][1]

if proba >= 0.6:
    st.error(f"🔴 ALTO RIESGO – Probabilidad de falla: {proba:.2f}")
elif proba >= 0.3:
    st.warning(f"🟠 PRECAUCIÓN – Probabilidad de falla: {proba:.2f}")
else:
    st.success(f"🟢 OPERACIÓN NORMAL – Probabilidad de falla: {proba:.2f}")

st.success(
    "Recomendación: **ajusta primero TORQUE**, luego desgaste y velocidad. "
    "La temperatura solo contextualiza."
)

# ======================================
# SECCIÓN ACADÉMICA (OCULTA)
# ======================================
with st.expander("📘 Sección académica (evaluación)"):
    st.markdown("""
**Modelo**
- Clasificador supervisado (Random Forest).
- Entrenado con datos etiquetados (AI4I 2020).

**Notas**
- Temperatura usada como variable contextual.
- El sistema prioriza recall de fallas.
""")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", round(metrics["accuracy"], 3))
    col2.metric("Recall (Falla)", round(metrics["recall_failure"], 3))

    st.subheader("Matriz de confusión")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(metrics["confusion_matrix"], cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, metrics["confusion_matrix"][i][j], ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Falla", "Falla"])
    ax.set_yticklabels(["No Falla", "Falla"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

st.caption("App operativa. Diseñada para apoyo a decisión, no para control automático.")
