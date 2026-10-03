import pytest

# src/models/promote_model.py sigue siendo un stub del andamiaje inicial (raise
# NotImplementedError) - hoy los candidatos (_oof, _provider_split) se guardan con sufijo y
# no hay promoción automatizada. Skip explícito; reemplazar cuando se implemente el módulo.
pytestmark = pytest.mark.skip(reason="src/models/promote_model.py todavía no está implementado (stub)")


def test_promote_model():
    raise NotImplementedError
