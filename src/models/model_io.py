"""Serialización y carga de modelos entrenados en models/."""

from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: Path) -> None:
    """Guarda un modelo entrenado (u otro objeto picklable, p.ej. artefactos de
    preprocesamiento) en disco vía joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Carga un modelo (u otro objeto) previamente guardado con save_model."""
    return joblib.load(Path(path))
