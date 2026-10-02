import pytest

# src/models/tune_model.py (búsqueda de hiperparámetros) sigue siendo un stub del
# andamiaje inicial (raise NotImplementedError). Skip explícito en vez de un test que
# "pase" sin probar nada; reemplazar cuando se implemente el módulo.
pytestmark = pytest.mark.skip(reason="src/models/tune_model.py todavía no está implementado (stub)")


def test_tune_model():
    raise NotImplementedError
