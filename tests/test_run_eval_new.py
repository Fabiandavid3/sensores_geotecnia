# tests/test_run_eval.py con nuevos criterios

import mlflow
import pytest

CRITERIOS = ["correctness", "relevance", "coherence", "toxicity", "harmfulness"]

@pytest.mark.parametrize("criterio", CRITERIOS)
def test_criterio_minimo(criterio):
    client = mlflow.tracking.MlflowClient()
    experiments = [e for e in client.search_experiments() if e.name.startswith("eval_")]

    assert experiments, "No hay experimentos con nombre 'eval_'"

    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert runs, f"No hay ejecuciones en el experimento {exp.name}"

        metric_key = f"{criterio}_score"
        scores = [r.data.metrics.get(metric_key, 0) for r in runs if metric_key in r.data.metrics]

        if scores:
            promedio = sum(scores) / len(scores)
            print(f"\n🔍 Detalles de '{criterio}' en {exp.name}:")

            for r in runs:
                score = r.data.metrics.get(metric_key)
                if score is not None:
                    pregunta = r.data.params.get("question", "Pregunta no encontrada")[:60]
                    print(f"  - Score: {score:.2f} | Pregunta: {pregunta}")

            print(f"📊 Promedio total de {criterio}: {promedio:.2f}")
            assert promedio >= 0.6, f"{criterio} insuficiente en {exp.name}: {promedio:.2f}"
