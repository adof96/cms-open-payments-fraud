"""Ingeniería de características a partir de los datos limpios de pagos."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera las features usadas por los modelos supervisado y no supervisado."""
    raise NotImplementedError
