"""Interfaz de inferencia para producir predicciones de fraude con un modelo entrenado."""

import pandas as pd


class Predictor:
    """Envuelve un modelo entrenado (supervisado o no supervisado) para producir scores de fraude."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
