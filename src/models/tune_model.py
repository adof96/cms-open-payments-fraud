"""Búsqueda de hiperparámetros para los modelos supervisado y no supervisado."""

import pandas as pd


def tune_model(X: pd.DataFrame, y: pd.Series | None = None, model_type: str = "supervised"):
    """Ejecuta la búsqueda de hiperparámetros para el tipo de modelo indicado."""
    raise NotImplementedError
