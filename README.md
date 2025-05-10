# 🧠 Chatbot RAG para Monitoreo Geotécnico

Este proyecto implementa un sistema de Recuperación Aumentada por Generación (RAG) para responder preguntas técnicas sobre monitoreo geotécnico en Colombia, usando LangChain, OpenAI, FAISS, Streamlit y MLflow. Está desarrollado como solución al desafío educativo propuesto en PyCon 2025.

---

## 🔗 Basado en

> Todo el proyecto fue adaptado y extendido a partir del repositorio original:  
> [GenAIOps_Pycon2025](https://github.com/darkanita/GenAIOps_Pycon2025/tree/main)  
>  
> 📖 El paso a paso detallado para su ejecución y estructura base se encuentra documentado allí.

---

## 🎓 Desafío educativo - Solución desarrollada

### 🧩 Parte 1: Personalización del dominio

- **Dominio elegido**: Monitoreo geotécnico en obras de infraestructura.
- **PDFs cargados**: Documentos técnicos reales de sensores, informes de taludes y manuales de calidad.
- **Prompts personalizados**:
  - `v1_asistente_geotecnia`
  - `v2_resumido_directo`
  - `v3_directo_flexible`
- **Dataset de evaluación**: `tests/eval_dataset.json`

📸 *Recomendación de imagen:* Captura de `data/pdfs/` y los prompts en `app/prompts/`

![data/pdfs/](img\data-pdfs.png)

![app/prompts/](img\promptv3.png)

---

### ✅ Parte 2: Evaluación automática con LangChain

- Implementado en `app/run_eval_new.py`
- Métricas:
  - `correctness_score`
  - `relevance_score`
  - `coherence_score`
  - `toxicity_score`
  - `harmfulness_score`
  - `qa_score` (binaria)
  - `clarity_score` (Bonus)

📸 *Recomendación:* Imagen de los resultados en MLflow con todas las métricas

---

### 👨‍🔬 Parte 3: Evaluación extendida

- Uso de `LabeledCriteriaEvalChain` con razonamientos.
- Cada criterio registra:
  - `score` como métrica
  - Razonamiento como parámetro truncado
  - Archivo `.txt` como artifact

📸 *Recomendación:* Vista de artifact subido (`coherence_reasoning_11.txt`)

---

### 📊 Parte 4: Dashboard mejorado

- Streamlit (`dashboard.py`) visualiza:
  - Runs por configuración
  - Radar por pregunta
  - Comparaciones por métrica
  - Razonamientos por pregunta/criterio

📸 *Recomendación:* Captura del radar + barra comparativa

---

### 🎤 Parte 5: Reflexión y comparación

- La mejor configuración fue `v3_directo_flexible` con chunks de 512.
- La versión `v1` mostró fallos en claridad y relevancia.
- Los razonamientos generados permitieron analizar las causas de los errores.

📸 *Recomendación:* Imagen del dashboard con tabla y razonamientos desplegados

---

### 🚀 BONUS: Nuevo criterio "claridad"

- Se integró el criterio `clarity_score` como evaluación adicional.
- Evaluación: *¿La respuesta es clara, fácil de entender y libre de ambigüedad?*

📸 *Recomendación:* Imagen del `clarity_score` y razonamiento generado.

---

## 🧪 Cómo ejecutar el proyecto

### 1. Instala dependencias

```bash
pip install -r requirements.txt