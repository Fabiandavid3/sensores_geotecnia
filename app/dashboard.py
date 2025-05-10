import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
import os

st.set_page_config(layout="wide")
st.title("📊 Dashboard de evaluación del chatbot geotécnico")

client = MlflowClient()

metricas = [
    "correctness_score", "relevance_score", "coherence_score",
    "toxicity_score", "harmfulness_score", "qa_score", "clarity_score"
]

experimentos = [e for e in client.search_experiments() if e.name.startswith("eval_")]
exp_names = [e.name for e in experimentos]

if not exp_names:
    st.error("❌ No hay experimentos 'eval_' en MLflow.")
    st.stop()

selected_exp = st.selectbox("Selecciona el experimento", exp_names)
experiment = next(e for e in experimentos if e.name == selected_exp)

runs = client.search_runs(experiment_ids=[experiment.experiment_id])
data = []

for run in runs:
    run_id = run.info.run_id
    metrics = run.data.metrics
    params = run.data.params

    row = {
        "run_id": run_id,
        "pregunta": params.get("question", ""),
        "prompt_version": params.get("prompt_version", ""),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0)),
    }

    for met in metricas:
        row[met] = metrics.get(met, None)
        razonamiento = params.get(f"{met}_reasoning")
        if razonamiento:
            row[f"{met}_razonamiento"] = razonamiento
        else:
            try:
                # Validar que el run tiene artifacts
                if client.list_artifacts(run_id):
                    artifacts_dir = download_artifacts(run_id)
                    for root, _, files in os.walk(artifacts_dir):
                        for file in files:
                            if file.startswith(f"{met}_reasoning") and file.endswith(".txt"):
                                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                                    row[f"{met}_razonamiento"] = f.read().strip()
                                break
                else:
                    print(f"⚠️ No hay artifacts registrados para run: {run_id}")
                    row[f"{met}_razonamiento"] = None
            except Exception as e:
                print(f"❌ No se pudo leer razonamiento para {met} en {run_id}: {e}")
                row[f"{met}_razonamiento"] = None

    data.append(row)

df = pd.DataFrame(data)

if df.empty:
    st.warning("⚠️ No se encontraron ejecuciones con métricas.")
    st.stop()

# 📋 Tabla general
st.subheader("📋 Resultados por ejecución")
st.dataframe(df[["run_id", "pregunta", "prompt_version", "chunk_size"] + metricas])

# 📊 Promedios por configuración
st.subheader("📊 Promedios por configuración")
agrupado = df.groupby(["prompt_version", "chunk_size"]).agg({
    met: "mean" for met in metricas
}).reset_index()
agrupado["config"] = agrupado["prompt_version"] + " | " + agrupado["chunk_size"].astype(str)
st.dataframe(agrupado)

# 📈 Comparación por métrica
st.subheader("📈 Comparación de configuraciones por métrica")
metrica_sel = st.selectbox("Selecciona la métrica:", metricas)
st.bar_chart(agrupado.set_index("config")[[metrica_sel]])

# 🕸 Radar por pregunta
st.subheader("📌 Comparación de criterios por pregunta")
pregunta_sel = st.selectbox("Selecciona una pregunta:", df["pregunta"].unique())
row_pregunta = df[df["pregunta"] == pregunta_sel].iloc[0]
radar_labels = ["correctness_score", "relevance_score", "coherence_score", "toxicity_score", "harmfulness_score", "qa_score", "clarity_score"]
radar_values = [row_pregunta.get(m, 0) for m in radar_labels]

fig = go.Figure(data=go.Scatterpolar(
    r=radar_values,
    theta=radar_labels,
    fill='toself',
    name='Puntajes'
))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=False,
    title="Radar de criterios para la pregunta seleccionada"
)
st.plotly_chart(fig, use_container_width=True)

# 🧠 Razonamientos
st.subheader("🧠 Razonamientos registrados")
for i, row in df.iterrows():
    st.markdown(f"**❓ Pregunta:** {row['pregunta']}")
    for met in metricas:
        razonamiento = row.get(f"{met}_razonamiento")
        if razonamiento:
            st.markdown(f"- **{met}:** {razonamiento}")
