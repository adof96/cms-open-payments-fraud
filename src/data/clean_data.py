"""Limpieza de datos y construcción de la etiqueta de fraude vía el cruce Open Payments x LEIE."""

import pandas as pd

from src.config import TARGET_COLUMN


def clean_open_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos, nulos y duplicados del dataset de Open Payments."""
    raise NotImplementedError


def clean_leie(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos, nulos y duplicados del dataset LEIE."""
    raise NotImplementedError


def label_fraud(payments_df: pd.DataFrame, leie_df: pd.DataFrame) -> pd.DataFrame:
    """Cruza Open Payments con LEIE y agrega la columna objetivo TARGET_COLUMN."""
    raise NotImplementedError
