import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# Configuración
# ======================================
st.set_page_config(
    page_title="Mantenimiento Predictivo",
    layout="centered"
)

# ======================================
# Carga de artefactos
# ======================================
model = joblib.load("artifacts/model_clf_solemne1.pkl")

with open("artifacts/metrics_solemne1.json", "r") as f:
    metrics = json.load(f)

df = pd.read_excel("artifacts/dataset_final_solemne1.xlsx")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

means = df[features].mean()
stds = df[features].std()
mins = df[features].min()
maxs = df[features].max()

# ======================================
# TÍTULO
# ======================================
st.title("Simulador de Mantenimiento Predictivo")

st.markdown("""
Evalúa un **caso operacional** y obtén una **recomendación práctica**
para reducir el riesgo de falla, basada en el comportamiento histórico del sistema.
""")

st.divider()

# ======================================
# 1️⃣ SIMULADOR (PRIMERO)
# ======================================
st.header("🔧 Simulación de un caso operativo")

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

# Umbral
threshold = st.slider(
    "Umbral de decisión (riesgo de falla)",
    0.3, 0.7, 0.5, 0.05
)

# ======================================
# 2️⃣ RESULTADO + RECOMENDACIÓN
# ======================================
if st.button("Evaluar riesgo"):
    input_data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }

    proba = model.predict_proba(pd.DataFrame([input_data]))[0][1]
    pred = proba >= threshold

    st.subheader("📊 Resultado del modelo")

    if pred:
        st.error(f"⚠️ Riesgo de FALLA\n\nProbabilidad estimada: {round(proba,3)}")
    else:
        st.success(f"✅ Operación dentro de rango normal\n\nProbabilidad estimada: {round(proba,3)}")

    # ----------------------------------
    # Recomendaciones operativas
    # ----------------------------------
    st.subheader("🛠️ Recomendación operativa")

    recommendations = []

    if torque > means["Torque [Nm]"] + stds["Torque [Nm]"]:
        recommendations.append("🔻 **Reducir torque** hacia valores típicos del sistema.")
    elif torque < means["Torque [Nm]"] - stds["Torque [Nm]"]:
        recommendations.append("🔺 **Aumentar torque** hacia el rango operativo normal.")

    if tool_wear > means["Tool wear [min]"]:
        recommendations.append("🔁 **Planificar mantenimiento** para reducir desgaste de herramienta.")

    if rot_speed < means["Rotational speed [rpm]"] - stds["Rotational speed [rpm]"]:
        recommendations.append("🔺 **Aumentar velocidad de rotación** hacia niveles normales.")
    elif rot_speed > means["Rotational speed [rpm]"] + stds["Rotational speed [rpm]"]:
        recommendations.append("🔻 **Reducir velocidad de rotación** para estabilizar operación.")

    if recommendations:
        for r in recommendations:
            st.write(r)
    else:
        st.write("✔ No se detectan desviaciones relevantes respecto al comportamiento normal del sistema.")

    # ==================================
    # 3️⃣ VISUALIZACIÓN DE POSICIÓN
    # ==================================
    st.subheader("📍 Posición del caso evaluado en el dataset")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.scatterplot(
        data=df,
        x="Tool wear [min]",
        y="Torque [Nm]",
        hue="Machine failure",
        alpha=0.3,
        ax=axs[0],
        legend=False
    )
    axs[0].scatter(tool_wear, torque, color="red", s=120, edgecolor="black")
    axs[0].set_title("Tool wear vs Torque")

    sns.scatterplot(
        data=df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue="Machine failure",
        alpha=0.3,
        ax=axs[1],
        legend=False
    )
    axs[1].scatter(rot_speed, torque, color="red", s=120, edgecolor="black")
    axs[1].set_title("Rotational speed vs Torque")

    st.pyplot(fig)

# ======================================
# 4️⃣ BLOQUE TÉCNICO DESPLEGABLE
# ======================================
with st.expander("📘 Detalle técnico del modelo (opcional)"):
    st.markdown("""
    **Modelo**
    - Clasificador supervisado entrenado en la Solemne 1
    - Dataset: AI4I 2020 Predictive Maintenance

    **Métricas**
    """)
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
        ax=ax
    )
    st.pyplot(fig)

    st.info("""
    Esta visualización es **exploratoria** y no implica relaciones causales.
    El objetivo es apoyar la toma de decisiones operativas.
    """)

# ======================================
# Cierre
# ======================================
st.caption("""
Aplicación desarrollada en Streamlit.  
Modelo entrenado y evaluado en la Solemne 1.  
Resultados reproducidos sin ajustes posteriores.
""")
