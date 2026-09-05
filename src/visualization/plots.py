"""Visualizaciones para exploración de datos y comparación de modelos."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_distribution(df: pd.DataFrame, column: str):
    """Grafica la distribución de una feature, comparando casos excluidos vs. no excluidos."""
    raise NotImplementedError


def plot_model_comparison(results: dict):
    """Grafica la comparación de métricas entre el enfoque supervisado y el no supervisado."""
    raise NotImplementedError


def plot_precision_recall_curves(curves: dict, save_path: Path) -> None:
    """Grafica una o más curvas precision-recall en un mismo gráfico y la guarda a disco.

    `curves` es {nombre_modelo: (recall, precision)} - p.ej. la salida de
    sklearn.metrics.precision_recall_curve por modelo (usada por
    src/models/tune_threshold.py para comparar Random Forest y XGBoost).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (recall, precision) in curves.items():
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — is_excluded (test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
