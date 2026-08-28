"""Evaluación y comparación de los enfoques supervisado y no supervisado."""

import pandas as pd


def evaluate_supervised(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de evaluación para el modelo supervisado."""
    raise NotImplementedError


def evaluate_unsupervised(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas de evaluación para el modelo no supervisado, usando la etiqueta solo para validar."""
    raise NotImplementedError
