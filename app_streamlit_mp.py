import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Configuración
# =========================================================
st.set_page_config(
    page_title="Simulador Operativo – Mantenimiento Predictivo",
    layout="centered"
)

MACHINE_LABEL = "Máquina industrial genérica (AI4I 2020 – Predictive Maintenance)"

# Orden EXACTO de features como se entrenó el modelo (CRÍTICO)
MODEL_FEATURE_ORDER = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# =========================================================
# Carga de artefactos
# =========================================================
model = joblib.load("artifacts/model_clf_solemne1.pkl")

with open("artifacts/metrics_solemne1.json", "r") as f:
    metrics = json.load(f)

df = pd.read_excel("artifacts/dataset_final_solemne1.xlsx")

# Stats para tablero y rangos
mins = df[MODEL_FEATURE_ORDER].min()
maxs = df[MODEL_FEATURE_ORDER].max()
means = df[MODEL_FEATURE_ORDER].mean()
pct = df[MODEL_FEATURE_ORDER].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).T

# =========================================================
# Helpers operativos
# =========================================================
def status_from_percentiles(var_name: str, value: float):
    """
    Semáforo operacional basado en percentiles del dataset.
    NORMAL: P25–P75
    ALERTA: (P10–P25) o (P75–P90)
    CRÍTICO: <P10 o >P90
    """
    p10, p25, p50, p75, p90 = [float(pct.loc[var_name, q]) for q in [0.10, 0.25, 0.50, 0.75, 0.90]]

    if value < p10 or value > p90:
        return "CRÍTICO", "🔴"
    if value < p25 or value > p75:
        return "ALERTA", "🟠"
    return "NORMAL", "🟢"

def action_direction(var_name: str, value: float):
    """
    Acción direccional conservadora hacia P50 (mediana del dataset).
    Tool wear se transforma a acción de mantención si está sobre mediana.
    """
    p50 = float(pct.loc[var_name, 0.50])

    if var_name == "Tool wear [min]":
        if value > p50:
            return "🔁 Planificar mantención"
        return "— Mantener"

    if value > p50:
        return "⬇ Disminuir"
    if value < p50:
        return "⬆ Aumentar"
    return "— Mantener"

def draw_compact_gauge(var_name: str, value: float, unit: str, contextual: bool = False):
    """
    Barra compacta tipo tablero:
    - Zonas por percentiles (rojo/ámbar/verde/ámbar/rojo)
    - Punto negro = valor actual
    - Línea discontinua = P50
    """
    p10, p25, p50, p75, p90 = [float(pct.loc[var_name, q]) for q in [0.10, 0.25, 0.50, 0.75, 0.90]]
    lo = float(mins[var_name])
    hi = float(maxs[var_name])
    if hi <= lo:
        hi = lo + 1.0

    status, icon = status_from_percentiles(var_name, value)
    direction = action_direction(var_name, value)

    fig, ax = plt.subplots(figsize=(5.3, 0.55))

    # Zonas (colores tipo PPT: verde suave + ámbar + rojo)
    ax.axvspan(lo, p10, color="#ffcccc", alpha=0.75)
    ax.axvspan(p10, p25, color="#ffe5b4", alpha=0.85)
    ax.axvspan(p25, p75, color="#d7f5d7", alpha=0.95)
    ax.axvspan(p75, p90, color="#ffe5b4", alpha=0.85)
    ax.axvspan(p90, hi, color="#ffcccc", alpha=0.75)

    # Mediana y punto del caso
    ax.axvline(p50, color="#1f77b4", linestyle="--", linewidth=1)  # azul sobrio
    ax.scatter([value], [0.5], s=90, color="black", zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel("")

    # Título compacto
    label = var_name.replace("[K]", "[°C]").replace("temperature", "temp")
    suffix = " (contextual)" if contextual else ""
    ax.set_title(f"{icon} {label}{suffix} | {status} | {direction}", loc="left", fontsize=9)

    # ticks mínimos para referencia
    ax.set_xticks([p25, p50, p75])
    ax.set_xticklabels([f"P25", f"P50", f"P75"], fontsize=7)

    plt.tight_layout(pad=0.35)
    return fig, status

def compute_priority(input_k: dict):
    """
    Prioridad operacional:
    - torque, desgaste, velocidad primero
    - temperaturas penalizadas (peso menor) para que no dominen el ranking
    """
    weights = {
        "Torque [Nm]": 1.0,
        "Tool wear [min]": 1.0,
        "Rotational speed [rpm]": 0.9,
        "Process temperature [K]": 0.4,
        "Air temperature [K]": 0.4
    }

    rows = []
    for v in MODEL_FEATURE_ORDER:
        sd = float(df[v].std()) if float(df[v].std()) > 0 else 1.0
        z = abs((float(input_k[v]) - float(means[v])) / sd)
        rows.append((v, weights.get(v, 1.0) * z))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def to_c(k):  # Kelvin -> Celsius
    return float(k) - 273.15

def to_k(c):  # Celsius -> Kelvin
    return float(c) + 273.15

# =========================================================
# UI – Cabecera
# =========================================================
st.title("Simulador Operativo de Mantenimiento Predictivo")
st.caption(MACHINE_LABEL)

st.info(
    "Uso operativo: mira el **tablero superior** (semáforo por variable). "
    "Ajusta primero **Torque**, luego **Desgaste** y **Velocidad**. "
    "Temperatura es **contextual**."
)

# =========================================================
# Inputs (operativos primero, temperatura al final en °C)
# =========================================================
st.header("🔧 Simulación (ajuste de parámetros)")

# Mostrar rangos operativos (mín–máx) clave
st.markdown(
    f"""
**Rangos del dataset (mín–máx):**
- Torque: **{mins['Torque [Nm]']:.1f}–{maxs['Torque [Nm]']:.1f} Nm**
- Tool wear: **{int(mins['Tool wear [min]'])}–{int(maxs['Tool wear [min]'])} min**
- Rotational speed: **{int(mins['Rotational speed [rpm]'])}–{int(maxs['Rotational speed [rpm]'])} rpm**
- Air temp: **{to_c(mins['Air temperature [K]']):.1f}–{to_c(maxs['Air temperature [K]']):.1f} °C**
- Process temp: **{to_c(mins['Process temperature [K]']):.1f}–{to_c(maxs['Process temperature [K]']):.1f} °C**
"""
)

col1, col2, col3 = st.columns(3)

with col1:
    torque = st.number_input(
        "Torque [Nm] (PRIORIDAD)",
        float(mins["Torque [Nm]"]),
        float(maxs["Torque [Nm]"]),
        float(means["Torque [Nm]"])
    )

with col2:
    tool_wear = st.number_input(
        "Tool wear [min]",
        int(mins["Tool wear [min]"]),
        int(maxs["Tool wear [min]"]),
        int(means["Tool wear [min]"])
    )

with col3:
    rot_speed = st.number_input(
        "Rotational speed [rpm]",
        int(mins["Rotational speed [rpm]"]),
        int(maxs["Rotational speed [rpm]"]),
        int(means["Rotational speed [rpm]"])
    )

st.markdown("### 🌡️ Contexto térmico (secundario)")

tcol1, tcol2 = st.columns(2)
with tcol1:
    air_c = st.number_input(
        "Air temperature [°C]",
        float(to_c(mins["Air temperature [K]"])),
        float(to_c(maxs["Air temperature [K]"])),
        float(to_c(means["Air temperature [K]"]))
    )
with tcol2:
    proc_c = st.number_input(
        "Process temperature [°C]",
        float(to_c(mins["Process temperature [K]"])),
        float(to_c(maxs["Process temperature [K]"])),
        float(to_c(means["Process temperature [K]"]))
    )

threshold = st.slider("Umbral de decisión (riesgo de falla)", 0.3, 0.7, 0.5, 0.05)

# Input en Kelvin para el modelo
input_k = {
    "Air temperature [K]": to_k(air_c),
    "Process temperature [K]": to_k(proc_c),
    "Rotational speed [rpm]": int(rot_speed),
    "Torque [Nm]": float(torque),
    "Tool wear [min]": int(tool_wear)
}

# =========================================================
# TABLERO SUPERIOR (compacto, una página)
# =========================================================
st.divider()
st.header("🚦 Tablero rápido (qué está peor)")

# Render gauges (compactos)
# Nota: para visual, dejamos temperaturas al final y marcadas como contextuales
dashboard_order = [
    ("Torque [Nm]", "Nm", False),
    ("Tool wear [min]", "min", False),
    ("Rotational speed [rpm]", "rpm", False),
    ("Process temperature [K]", "°C", True),
    ("Air temperature [K]", "°C", True),
]

for var, unit, contextual in dashboard_order:
    val = float(input_k[var])
    # Para mostrar en °C en tablero, convertimos el valor de K a C SOLO para lectura
    display_val = to_c(val) if unit == "°C" else val
    fig, _ = draw_compact_gauge(var, display_val, unit, contextual=contextual)
    st.pyplot(fig)

# =========================================================
# Evaluación de riesgo
# =========================================================
st.divider()
st.header("📊 Evaluación de riesgo")

# Forzar orden exacto de columnas (FIX definitivo del error)
X_input = pd.DataFrame([input_k])[MODEL_FEATURE_ORDER]
proba = float(model.predict_proba(X_input)[0][1])

if proba >= 0.60:
    st.error(f"🔴 ALTO RIESGO – Probabilidad de falla: {proba:.3f}")
elif proba >= 0.30:
    st.warning(f"🟠 PRECAUCIÓN – Probabilidad de falla: {proba:.3f}")
else:
    st.success(f"🟢 OPERACIÓN NORMAL – Probabilidad de falla: {proba:.3f}")

# Prioridad de intervención
st.subheader("🎯 Prioridad de intervención (operación)")
priority = compute_priority(input_k)
top_var, top_score = priority[0]
status, icon = status_from_percentiles(top_var, float(input_k[top_var]))
direction = action_direction(top_var, float(input_k[top_var]))

# Mostrar ranking compacto
pr_df = pd.DataFrame(priority, columns=["Variable", "Prioridad (score)"])
pr_df["Prioridad (score)"] = pr_df["Prioridad (score)"].round(2)
st.dataframe(pr_df, use_container_width=True, hide_index=True)

st.success(f"Acción recomendada #1: **{top_var}** → {icon} **{status}** → **{direction}**")

st.caption("Regla operativa: corrige primero torque; luego desgaste y velocidad. Temperatura contextualiza.")

# =========================================================
# Sección académica (oculta)
# =========================================================
with st.expander("📘 Sección académica (para profesor / evaluación)"):
    st.markdown("""
**Modelo**
- Clasificador supervisado (**Random Forest**) entrenado con datos etiquetados para predecir fallas operacionales.

**Notas**
- El tablero operacional utiliza percentiles del dataset para clasificar NORMAL/ALERTA/CRÍTICO.
- La temperatura se trata como variable **contextual** (no prioritaria).
- No implica causalidad: apoyo a decisión.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", round(metrics["accuracy"], 3))
    with c2:
        st.metric("Recall (Falla)", round(metrics["recall_failure"], 3))

    st.subheader("Matriz de confusión")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Falla", "Falla"])
    ax.set_yticklabels(["No Falla", "Falla"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

st.caption("App operativa en Streamlit. Diseñada para apoyo a decisión, no control automático.")
