"""Visualizaciones para exploración de datos y comparación de modelos."""

import pandas as pd


def plot_feature_distribution(df: pd.DataFrame, column: str):
    """Grafica la distribución de una feature, comparando casos excluidos vs. no excluidos."""
    raise NotImplementedError


def plot_model_comparison(results: dict):
    """Grafica la comparación de métricas entre el enfoque supervisado y el no supervisado."""
    raise NotImplementedError
