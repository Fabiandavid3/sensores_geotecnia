# 🧠 Chatbot RAG para Monitoreo Geotécnico

Este proyecto implementa un sistema de Recuperación Aumentada por Generación (RAG) que responde preguntas sobre documentos técnicos relacionados con sensores, taludes y movimientos en masa en Colombia. Utiliza LangChain, OpenAI, FAISS, Streamlit y MLflow para ingesta, vectorización, generación de respuestas y evaluación automática.

---

## 🎓 Desafío educativo - Solución desarrollada

### 🧩 Parte 1: Personalización del dominio

- **Dominio elegido**: Monitoreo geotécnico en proyectos de infraestructura.
- **Fuentes**: Documentos PDF de campo sobre sensores, monitoreo de taludes y manuales de calidad.
- **Ubicación de los documentos**: `data/pdfs/`
- **Prompts personalizados**: Incluidos en `app/prompts/` (versiones `v1_asistente_geotecnia`, `v2_resumido`, `v3_directo_flexible`).
- **Conjunto de evaluación**: `tests/eval_dataset.json` con 13 preguntas específicas y sus respuestas de referencia.

📸 *Sugerencia de imagen:* Captura de `data/pdfs/` y estructura de prompts (`app/prompts/`)

---

### ✅ Parte 2: Evaluación automática con LangChain

- Script `app/run_eval_new.py` ejecuta la evaluación automática.
- Evalúa con `LabeledCriteriaEvalChain` y `QAEvalChain` de LangChain.
- Métricas generadas:
  - `correctness_score`
  - `relevance_score`
  - `coherence_score`
  - `toxicity_score`
  - `harmfulness_score`
  - `clarity_score` (Bonus)
  - `qa_score` (binaria)

📸 *Sugerencia de imagen:* Vista en MLflow mostrando métricas de una run (`eval_q11`)

---

### 👨‍🔬 Parte 3: Sistema de evaluación extendido

- Se integró evaluación multicriterio con razonamientos generados por el LLM.
- Cada criterio registra:
  - Score (como métrica)
  - Texto razonado (como parámetro y artifact `.txt`)

📸 *Sugerencia de imagen:* Artifact cargado en MLflow con razonamiento (`harmfulness_reasoning_11.txt`)

---

### 📊 Parte 4: Mejora del dashboard

- Streamlit Dashboard (`dashboard.py`) permite:
  - Visualizar métricas por ejecución
  - Comparar configuraciones de chunk y prompt
  - Mostrar razonamientos generados
  - Comparar métricas por pregunta en gráfico radar

📸 *Sugerencia de imagen:* Radar de una pregunta + tabla general de métricas

---

### 🎤 Parte 5: Reflexión y comparaciones

- Se compararon versiones `v1_asistente_geotecnia`, `v2_resumido`, `v3_directo_flexible`.
- La versión `v3` obtuvo las puntuaciones más altas en claridad, coherencia y relevancia.
- Se observaron fallos frecuentes en `correctness` por ambigüedad en la extracción del contexto.
- Toxicidad y daño siempre puntuaron bajo (lo esperado).

📸 *Sugerencia de imagen:* Comparativa de configuraciones por métrica en gráfico de barras

---

### 🚀 Bonus: Nuevo criterio — "claridad"

- Se implementó un nuevo criterio: `clarity`
- Definición: *¿Es clara y fácil de entender la respuesta para un lector no experto?*
- Integrado en el pipeline de evaluación y visualización.

📸 *Sugerencia de imagen:* `clarity_score` en tabla de dashboard o ejemplo de razonamiento generado

---

## 🧪 Cómo ejecutar el proyecto

### 1. Instala dependencias

```bash
pip install -r requirements.txt
