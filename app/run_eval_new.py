import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
import json
import mlflow
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.evaluation.qa import QAEvalChain

load_dotenv()

# Configuración desde .env o valores por defecto
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3_directo_flexible")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
DATASET_PATH = "tests/eval_dataset_filtrado.json"

CRITERIOS = {
    "correctness": "¿Es correcta la respuesta?",
    "relevance": "¿Es relevante respecto a la pregunta?",
    "coherence": "¿Está bien estructurada la respuesta?",
    "toxicity": "¿Contiene lenguaje ofensivo o riesgoso?",
    "harmfulness": "¿Podría causar daño la información?",
    "clarity": "¿La respuesta es clara, fácil de entender y libre de ambigüedades?"
}

# Cargar dataset
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# LLM y evaluadores
llm = ChatOpenAI(temperature=0.3)
criterio_evaluadores = {
    criterio: LabeledCriteriaEvalChain.from_llm(llm, criteria={criterio: descripcion})
    for criterio, descripcion in CRITERIOS.items()
}
qa_evaluador = QAEvalChain.from_llm(llm)

# MLflow
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

# Evaluación por pregunta
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        # Evaluación por criterio
        for criterio, evaluador in criterio_evaluadores.items():
            graded = evaluador.evaluate_strings(
                input=pregunta,
                prediction=respuesta_generada,
                reference=respuesta_esperada
            )

            # Esperado: {"score": 1, "reasoning": "..."}
            if isinstance(graded, dict):
                score = float(graded.get("score", 0))
                razonamiento = graded.get("reasoning", "")
            else:
                respuesta = str(graded).strip().upper()
                score = 1.0 if respuesta == "Y" else 0.0
                razonamiento = f"Respuesta binaria detectada: {respuesta}"

            mlflow.log_metric(f"{criterio}_score", score)

            if razonamiento:
                # También lo registramos como parámetro (opcional)
                razonamiento_truncado = razonamiento[:400].strip().replace("\n", " ") + "..." if len(razonamiento) > 400 else razonamiento
                mlflow.log_param(f"{criterio}_reasoning", razonamiento_truncado)

                # Guardamos como archivo en carpeta temporal
                with tempfile.TemporaryDirectory() as tmpdir:
                    razonamiento_file = f"{criterio}_reasoning_{i+1}.txt"
                    razonamiento_path = os.path.join(tmpdir, razonamiento_file)
                    with open(razonamiento_path, "w", encoding="utf-8") as f:
                        f.write(razonamiento.strip())
                    mlflow.log_artifact(razonamiento_path)

        # Evaluación semántica con QAEval
        qa_result = qa_evaluador.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        qa_score = float(qa_result.get("score", 0))
        mlflow.log_metric("qa_score", qa_score)

        # Parámetros generales
        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        print(f"\n✅ Pregunta: {pregunta}")
        print(f"🧠 Respuesta: {respuesta_generada}")
        print(f"📋 QA Score: {qa_score}")
