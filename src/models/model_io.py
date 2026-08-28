"""Serialización y carga de modelos entrenados en models/."""

from pathlib import Path
from typing import Any


def save_model(model: Any, path: Path) -> None:
    """Guarda un modelo entrenado en disco."""
    raise NotImplementedError


def load_model(path: Path) -> Any:
    """Carga un modelo previamente guardado desde disco."""
    raise NotImplementedError
