import pytest

# src/models/evaluate_model.py sigue siendo un stub del andamiaje inicial (raise
# NotImplementedError): la evaluación real vive hoy en train_supervised._evaluate y
# train_unsupervised._evaluate_anomaly_ranking, ya cubiertas por sus tests. Skip explícito
# en vez de un test que "pase" sin probar nada; reemplazar cuando se implemente el módulo.
pytestmark = pytest.mark.skip(reason="src/models/evaluate_model.py todavía no está implementado (stub)")


def test_evaluate_supervised():
    raise NotImplementedError


def test_evaluate_unsupervised():
    raise NotImplementedError
