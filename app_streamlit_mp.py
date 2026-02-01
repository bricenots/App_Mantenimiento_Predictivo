import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======================================
# Configuración
# ======================================
st.set_page_config(page_title="Simulador Operativo – Mantenimiento Predictivo", layout="centered")

MACHINE_LABEL = "Máquina industrial (dataset AI4I 2020 – mantenimiento predictivo)"

# ======================================
# Carga de artefactos
# ======================================
model = joblib.load("artifacts/model_clf_solemne1.pkl")

with open("artifacts/metrics_solemne1.json", "r") as f:
    metrics = json.load(f)

# Dataset (para rangos, percentiles y visualización operativa)
df = pd.read_excel("artifacts/dataset_final_solemne1.xlsx")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Stats del dataset
mins = df[features].min()
maxs = df[features].max()
means = df[features].mean()
stds = df[features].std()

# Percentiles para semáforo operacional
pct = df[features].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).T
# pct[col] -> index = [0.10,0.25,0.50,0.75,0.90]

# ======================================
# Funciones utilitarias
# ======================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def status_and_action(var_name: str, value: float):
    """
    Clasificación operacional basada en percentiles del dataset.
    - Normal: P25–P75
    - Alerta: (P10–P25) o (P75–P90)
    - Crítico: <P10 o >P90
    Acción: direccional hacia la mediana (P50)
    """
    p10, p25, p50, p75, p90 = [float(pct.loc[var_name, q]) for q in [0.10, 0.25, 0.50, 0.75, 0.90]]

    if value < p10 or value > p90:
        status = "CRÍTICO"
        icon = "🔴"
    elif value < p25 or value > p75:
        status = "ALERTA"
        icon = "🟠"
    else:
        status = "NORMAL"
        icon = "🟢"

    # Acción direccional conservadora (acercar a P50)
    if value > p50:
        direction = "⬇ Disminuir"
    elif value < p50:
        direction = "⬆ Aumentar"
    else:
        direction = "— Mantener"

    # Ajuste de lenguaje para Tool wear: “planificar mantención”
    if var_name == "Tool wear [min]":
        if status in ["ALERTA", "CRÍTICO"] and value > p50:
            direction = "🔁 Planificar mantención"
        elif status == "NORMAL":
            direction = "— Mantener"

    return icon, status, direction, (p10, p25, p50, p75, p90)

def make_semaphore_plot(var_name: str, value: float):
    """
    Barra horizontal con zonas (verde/amarillo/rojo) + punto del caso.
    """
    icon, status, direction, (p10, p25, p50, p75, p90) = status_and_action(var_name, value)

    lo = float(mins[var_name])
    hi = float(maxs[var_name])

    # Para variables con rango muy estrecho (temperaturas), amplía un pelín para visualizar bien
    span = hi - lo
    if span == 0:
        hi = lo + 1.0
        span = 1.0

    fig, ax = plt.subplots(figsize=(6.5, 1.2))

    # Zonas por percentiles, recortadas a [lo, hi]
    a = clamp(p10, lo, hi)
    b = clamp(p25, lo, hi)
    c = clamp(p75, lo, hi)
    d = clamp(p90, lo, hi)

    # Segmentos
    ax.axvspan(lo, a, alpha=0.25)      # extremo bajo (rojo implícito)
    ax.axvspan(a, b, alpha=0.20)       # alerta baja
    ax.axvspan(b, c, alpha=0.18)       # normal
    ax.axvspan(c, d, alpha=0.20)       # alerta alta
    ax.axvspan(d, hi, alpha=0.25)      # extremo alto

    # Colores: fijo por patch order
    colors = ["#ffcccc", "#ffe5b4", "#d7f5d7", "#ffe5b4", "#ffcccc"]
    for patch, col in zip(ax.patches, colors):
        patch.set_facecolor(col)
        patch.set_edgecolor("none")

    # Punto del caso
    ax.scatter([value], [0.5], s=120, color="red", edgecolor="black", zorder=5)

    # Línea mediana
    ax.axvline(p50, linestyle="--", linewidth=1)

    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    ax.set_title(f"{icon} {var_name}  |  {status}  |  {direction}", fontsize=10)
    ax.set_xlabel("")

    # Ticks mínimos: p25, p50, p75
    ax.set_xticks([b, p50, c])
    ax.set_xticklabels([f"P25\n{b:.1f}", f"P50\n{p50:.1f}", f"P75\n{c:.1f}"], fontsize=8)

    plt.tight_layout()
    return fig

def compute_priority_scores(input_dict: dict):
    """
    Score por variable: distancia normalizada a la mediana (|z| aproximado).
    Usamos std del dataset para priorizar qué tocar primero.
    """
    scores = []
    for v in features:
        x = float(input_dict[v])
        sd = float(stds[v]) if float(stds[v]) > 0 else 1.0
        z = abs((x - float(means[v])) / sd)
        scores.append((v, z))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores

# ======================================
# UI – CABECERA
# ======================================
st.title("Simulador Operativo de Mantenimiento Predictivo")
st.caption(f"Aplicación enfocada en operación segura. {MACHINE_LABEL}")

st.markdown("""
Aquí simulas un caso operativo y obtienes:
- **Riesgo estimado** (Falla / Operación normal)
- **Qué variable tocar primero**
- **Dirección recomendada** (subir / bajar hacia rango normal)
- **Dónde estás** respecto al histórico del sistema
""")

st.divider()

# ======================================
# SECCIÓN OPERATIVA (PRIMERO)
# ======================================
st.header(f"🔧 Simulador operativo – {MACHINE_LABEL}")

# Rangos min/max en el título (como pediste)
st.markdown(
    f"""
**Rangos del dataset (mín–máx):**
- Torque: **{mins['Torque [Nm]']:.1f} – {maxs['Torque [Nm]']:.1f}** Nm  
- Tool wear: **{int(mins['Tool wear [min]'])} – {int(maxs['Tool wear [min]'])}** min  
- Rotational speed: **{int(mins['Rotational speed [rpm]'])} – {int(maxs['Rotational speed [rpm]'])}** rpm
"""
)

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air temperature [K]",
        float(mins["Air temperature [K]"]),
        float(maxs["Air temperature [K]"]),
        float(means["Air temperature [K]"])
    )
    proc_temp = st.number_input(
        "Process temperature [K]",
        float(mins["Process temperature [K]"]),
        float(maxs["Process temperature [K]"]),
        float(means["Process temperature [K]"])
    )
    rot_speed = st.number_input(
        "Rotational speed [rpm]",
        int(mins["Rotational speed [rpm]"]),
        int(maxs["Rotational speed [rpm]"]),
        int(means["Rotational speed [rpm]"])
    )

with col2:
    torque = st.number_input(
        "Torque [Nm]",
        float(mins["Torque [Nm]"]),
        float(maxs["Torque [Nm]"]),
        float(means["Torque [Nm]"])
    )
    tool_wear = st.number_input(
        "Tool wear [min]",
        int(mins["Tool wear [min]"]),
        int(maxs["Tool wear [min]"]),
        int(means["Tool wear [min]"])
    )

threshold = st.slider("Umbral de decisión (riesgo de falla)", 0.3, 0.7, 0.5, 0.05)

input_data = {
    "Air temperature [K]": float(air_temp),
    "Process temperature [K]": float(proc_temp),
    "Rotational speed [rpm]": int(rot_speed),
    "Torque [Nm]": float(torque),
    "Tool wear [min]": int(tool_wear)
}

# Evaluación
if st.button("Evaluar riesgo"):
    proba = model.predict_proba(pd.DataFrame([input_data]))[0][1]
    is_failure = (proba >= threshold)

    # 1) Estado global (semáforo grande)
    st.subheader("🚦 Estado global")
    if proba >= 0.60:
        st.error(f"🔴 ALTO RIESGO – Corregir antes de continuar | Probabilidad: {proba:.3f}")
    elif proba >= 0.30:
        st.warning(f"🟠 PRECAUCIÓN – Operar con control | Probabilidad: {proba:.3f}")
    else:
        st.success(f"🟢 OPERACIÓN NORMAL – Continuar | Probabilidad: {proba:.3f}")

    # 2) Ranking: qué tocar primero
    st.subheader("🎯 Qué tocar primero (prioridad operativa)")
    priority = compute_priority_scores(input_data)
    top_var, top_score = priority[0]

    pr_df = pd.DataFrame(priority, columns=["Variable", "Desviación (|z| aprox.)"])
    pr_df["Desviación (|z| aprox.)"] = pr_df["Desviación (|z| aprox.)"].round(2)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(pr_df["Variable"][::-1], pr_df["Desviación (|z| aprox.)"][::-1])
    ax.set_xlabel("Desviación respecto a operación típica")
    ax.set_ylabel("")
    ax.set_title("Ranking de intervención (más desviado = más prioridad)")
    st.pyplot(fig)

    icon, status, direction, _ = status_and_action(top_var, float(input_data[top_var]))
    st.info(f"Prioridad #1: **{top_var}** → {icon} **{status}** → **{direction}**")

    # 3) Semáforo por variable (operación segura)
    st.subheader("🟩🟧🟥 Semáforo por variable (zona segura)")
    for v in ["Torque [Nm]", "Tool wear [min]", "Rotational speed [rpm]", "Process temperature [K]", "Air temperature [K]"]:
        fig = make_semaphore_plot(v, float(input_data[v]))
        st.pyplot(fig)

    # 4) What-if (solo Torque, el control más “operable”)
    st.subheader("🔁 Simulación rápida (What-if) – Ajuste de Torque")
    torque_w = st.slider(
        "Probar Torque [Nm]",
        float(mins["Torque [Nm]"]),
        float(maxs["Torque [Nm]"]),
        float(input_data["Torque [Nm]"]),
        0.5
    )

    input_whatif = dict(input_data)
    input_whatif["Torque [Nm]"] = float(torque_w)

    proba_now = float(proba)
    proba_w = float(model.predict_proba(pd.DataFrame([input_whatif]))[0][1])

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Prob. actual", f"{proba_now:.3f}")
    with c2:
        st.metric("Prob. con Torque ajustado", f"{proba_w:.3f}")

    if proba_w < proba_now:
        st.success("✅ El ajuste propuesto reduce el riesgo estimado.")
    elif proba_w > proba_now:
        st.warning("🟠 El ajuste propuesto aumenta el riesgo estimado.")
    else:
        st.info("ℹ️ El ajuste no cambia el riesgo estimado de forma relevante.")

    # 5) Visualización de posición en el dataset (rápida, visual)
    st.subheader("📍 Dónde estás vs histórico (operación)")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.scatterplot(
        data=df,
        x="Tool wear [min]",
        y="Torque [Nm]",
        hue="Machine failure",
        alpha=0.25,
        ax=axs[0],
        legend=False
    )
    axs[0].scatter(input_data["Tool wear [min]"], input_data["Torque [Nm]"], color="red", s=140, edgecolor="black")
    axs[0].set_title("Tool wear vs Torque (punto = tu caso)")

    sns.scatterplot(
        data=df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue="Machine failure",
        alpha=0.25,
        ax=axs[1],
        legend=False
    )
    axs[1].scatter(input_data["Rotational speed [rpm]"], input_data["Torque [Nm]"], color="red", s=140, edgecolor="black")
    axs[1].set_title("Rotational speed vs Torque (punto = tu caso)")

    st.pyplot(fig)

st.divider()

# ======================================
# SECCIÓN ACADÉMICA (OCULTA)
# ======================================
with st.expander("📘 Sección académica (para profesor / evaluación)"):
    st.markdown("""
**Modelo**
- Clasificador supervisado (**Random Forest**) entrenado con datos etiquetados para predecir fallas operacionales.

**Notas**
- Las recomendaciones operativas se basan en la comparación con rangos típicos del dataset (percentiles).
- No se interpreta causalidad: es apoyo a decisión.
""")

    colA, colB = st.columns(2)
    with colA:
        st.metric("Accuracy", round(metrics["accuracy"], 3))
    with colB:
        st.metric("Recall (Falla)", round(metrics["recall_failure"], 3))

    st.subheader("Matriz de confusión")
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

    st.info("Este bloque técnico se mantiene oculto para priorizar una interfaz operativa simple.")

st.caption("Aplicación operativa en Streamlit. Resultados reproducidos desde artefactos de Solemne 1.")
