import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import numpy as np

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Simulador Operativo – Mantenimiento Predictivo",
    layout="centered"
)

MACHINE_LABEL = "Máquina industrial genérica – AI4I 2020"

MODEL_FEATURE_ORDER = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# =========================================================
# CARGA DE ARTEFACTOS
# =========================================================
model = joblib.load("artifacts/model_clf_solemne1.pkl")

with open("artifacts/metrics_solemne1.json", "r") as f:
    metrics = json.load(f)

df = pd.read_excel("artifacts/dataset_final_solemne1.xlsx")

mins = df[MODEL_FEATURE_ORDER].min()
maxs = df[MODEL_FEATURE_ORDER].max()
means = df[MODEL_FEATURE_ORDER].mean()
pct = df[MODEL_FEATURE_ORDER].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).T

# =========================================================
# HELPERS
# =========================================================
def to_c(k): return float(k) - 273.15
def to_k(c): return float(c) + 273.15

def draw_gauge(var, value, unit):
    p10, p25, p75, p90 = pct.loc[var, [0.10, 0.25, 0.75, 0.90]]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 26}},
        title={"text": var, "font": {"size": 14}},
        gauge={
            "axis": {"range": [float(mins[var]), float(maxs[var])]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [float(mins[var]), p10], "color": "#ff4d4d"},
                {"range": [p10, p25], "color": "#ffb347"},
                {"range": [p25, p75], "color": "#4CAF50"},
                {"range": [p75, p90], "color": "#ffb347"},
                {"range": [p90, float(maxs[var])], "color": "#ff4d4d"},
            ],
        }
    ))

    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# =========================================================
# CABECERA
# =========================================================
st.title("🛠️ Simulador Operativo de Mantenimiento Predictivo")
st.caption(MACHINE_LABEL)

st.info(
    "Mira los relojes. "
    "🔴 Rojo = corrige. 🟡 Amarillo = atención. 🟢 Verde = operar normal. "
    "Prioriza Torque → Desgaste → Velocidad."
)

# =========================================================
# VALORES INICIALES
# =========================================================
torque = float(means["Torque [Nm]"])
tool_wear = int(means["Tool wear [min]"])
rot_speed = int(means["Rotational speed [rpm]"])
air_c = float(to_c(means["Air temperature [K]"]))
proc_c = float(to_c(means["Process temperature [K]"]))

# =========================================================
# TABLERO RÁPIDO (SOLO LO IMPORTANTE)
# =========================================================
st.header("🚦 Estado actual")

c1, c2, c3 = st.columns(3)

with c1:
    st.plotly_chart(
        draw_gauge("Torque [Nm]", torque, "Nm"),
        use_container_width=True
    )

with c2:
    st.plotly_chart(
        draw_gauge("Tool wear [min]", tool_wear, "min"),
        use_container_width=True
    )

with c3:
    st.plotly_chart(
        draw_gauge("Rotational speed [rpm]", rot_speed, "rpm"),
        use_container_width=True
    )

# =========================================================
# SIMULADOR
# =========================================================
st.divider()
st.header("🔧 Ajuste de parámetros")

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

st.markdown("### 🌡️ Temperatura (contexto, no prioritaria)")

air_c = st.slider(
    "Air temperature [°C]",
    float(to_c(mins["Air temperature [K]"])),
    float(to_c(maxs["Air temperature [K]"])),
    air_c
)

proc_c = st.slider(
    "Process temperature [°C]",
    float(to_c(mins["Process temperature [K]"])),
    float(to_c(maxs["Process temperature [K]"])),
    proc_c
)

# =========================================================
# EVALUACIÓN
# =========================================================
input_k = {
    "Air temperature [K]": to_k(air_c),
    "Process temperature [K]": to_k(proc_c),
    "Rotational speed [rpm]": int(rot_speed),
    "Torque [Nm]": float(torque),
    "Tool wear [min]": int(tool_wear)
}

X_input = pd.DataFrame([input_k])[MODEL_FEATURE_ORDER]
proba = float(model.predict_proba(X_input)[0][1])

st.divider()
st.header("📊 Riesgo estimado")

if proba >= 0.6:
    st.error(f"🔴 ALTO RIESGO – Probabilidad de falla: {proba:.2f}")
elif proba >= 0.3:
    st.warning(f"🟡 PRECAUCIÓN – Probabilidad de falla: {proba:.2f}")
else:
    st.success(f"🟢 OPERACIÓN NORMAL – Probabilidad de falla: {proba:.2f}")

st.success(
    "Recomendación: si hay riesgo, **baja TORQUE** primero. "
    "Luego evalúa desgaste y velocidad. Temperatura solo contextualiza."
)

# =========================================================
# SECCIÓN ACADÉMICA (OCULTA)
# =========================================================
with st.expander("📘 Sección académica (evaluación)"):
    st.markdown("""
**Modelo**
- Clasificador supervisado (Random Forest).
- Entrenado con AI4I 2020.

**Notas**
- Temperatura usada como variable contextual.
- La app prioriza decisión operativa.
""")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", round(metrics["accuracy"], 3))
    c2.metric("Recall (Falla)", round(metrics["recall_failure"], 3))

    st.write("Matriz de confusión:", metrics["confusion_matrix"])

st.caption("App operativa – apoyo a decisión, no control automático.")
