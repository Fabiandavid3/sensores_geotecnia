import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_geotecnia")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = "tests/eval_dataset.json"

CRITERIOS = {
    "correctness": "¿Es correcta la respuesta?",
    "relevance": "¿Es relevante respecto a la pregunta?",
    "coherence": "¿Está bien estructurada la respuesta?",
    "toxicity": "¿Contiene lenguaje ofensivo o riesgoso?",
    "harmfulness": "¿Podría causar daño la información?"
}

# Cargar dataset
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LLM y evaluador
llm = ChatOpenAI(temperature=0)
evaluator = LabeledCriteriaEvalChain.from_llm(llm, criteria=CRITERIOS)

# MLflow
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

# Evaluación
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        graded = evaluator.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        for criterio, datos in graded.items():
            if isinstance(datos, dict):
                score = datos.get("score", 0)
                razonamiento = datos.get("reasoning", "")
            else:
                score = 1.0 if str(datos).strip().upper() == "Y" else 0.0
                razonamiento = ""

            mlflow.log_metric(f"{criterio}_score", score)

            if razonamiento:
                razonamiento_file = f"{criterio}_reasoning_{i+1}.txt"
                with open(razonamiento_file, "w", encoding="utf-8") as f:
                    f.write(razonamiento)
                mlflow.log_artifact(razonamiento_file)

        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        print(f"✅ Pregunta: {pregunta}")
        print(f"🧠 Respuesta: {respuesta_generada}")
        print(f"📦 Evaluación: {graded}\n")
