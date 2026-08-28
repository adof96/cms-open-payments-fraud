"""Promoción del mejor modelo candidato a modelo de producción en models/."""

from pathlib import Path


def promote_model(candidate_path: Path, production_path: Path) -> None:
    """Reemplaza el modelo en producción por el candidato, tras validar sus métricas."""
    raise NotImplementedError
