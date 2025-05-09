import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator


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
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)


# Evaluador con criterios (estructura separada por criterio)
evaluator = load_evaluator("multi_criteria", llm=llm, criteria=list(CRITERIOS.keys()))

# Establecer experimento MLflow
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"\U0001f4ca Experimento MLflow: eval_{PROMPT_VERSION}")

for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        # Evaluar con los criterios definidos
        graded = evaluator.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        #Temporal
        import pprint
        print(f"\n🧪 Resultado estructurado del evaluador para pregunta {i+1}:")
        pprint.pprint(graded)

        print(f"\n\U0001f4e6 Resultado evaluación para pregunta {i+1}/{len(dataset)}:")
        print(graded)

        for criterio in CRITERIOS:
            criterio_data = graded.get(criterio, {})
            score = criterio_data.get("score", 0)
            razonamiento = criterio_data.get("reasoning", "")
            score_key = f"{criterio}_score"

            if razonamiento:
                razonamiento_file = f"{criterio}_reasoning_{i+1}.txt"
                with open(razonamiento_file, "w", encoding="utf-8") as f:
                    f.write(razonamiento)
                mlflow.log_artifact(razonamiento_file)

        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        print(f"\u2705 Pregunta: {pregunta}")
        print(f"\U0001f9e0 Respuesta: {respuesta_generada}")