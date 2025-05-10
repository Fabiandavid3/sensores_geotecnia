import mlflow
import pytest

# Lista de métricas que se espera evaluar
METRICAS_ESPERADAS = [
    "correctness_score",
    "relevance_score",
    "coherence_score",
    "toxicity_score",
    "harmfulness_score",
    "qa_score",
    "clarity_score"
]

UMBRAL_MINIMO = 0.7

@pytest.mark.parametrize("metrica", METRICAS_ESPERADAS)
def test_eval_scores_minimos(metrica):
    client = mlflow.tracking.MlflowClient()
    experiments = [e for e in client.search_experiments() if e.name == "eval_v3_directo_flexible"]

    assert experiments, "❌ No hay experimentos con nombre que empiece con 'eval_'"

    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert runs, f"❌ No hay ejecuciones en el experimento {exp.name}"

        scores = [r.data.metrics.get(metrica, None) for r in runs if metrica in r.data.metrics]

        if not scores:
            pytest.fail(f"❌ No se encontraron métricas '{metrica}' en {exp.name}")

        promedio = sum(scores) / len(scores)
        print(f"📈 Promedio de '{metrica}' en {exp.name}: {promedio:.2f}")
        assert promedio >= UMBRAL_MINIMO, f"⚠️ Precisión insuficiente para '{metrica}' en {exp.name}: {promedio:.2f}"
