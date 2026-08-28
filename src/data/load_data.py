"""Descarga y carga de los datasets crudos: CMS Open Payments (PY2024, General Payments)
y LEIE (List of Excluded Individuals/Entities).

Cada fuente se descarga a su propia subcarpeta de data/raw/ y se cachea localmente:
si el archivo ya existe, no se vuelve a descargar (usar force=True para forzar).
"""

import sys

import pandas as pd
import requests

from src.config import (
    CMS_OPEN_PAYMENTS_RAW_DIR,
    LEIE_FILENAME,
    LEIE_RAW_DIR,
    LEIE_URL,
    OPEN_PAYMENTS_FILENAME,
    OPEN_PAYMENTS_URL,
)

CHUNK_SIZE = 1024 * 1024  # 1 MB


def _download_with_cache(url: str, dest_path, force: bool = False):
    """Descarga `url` a `dest_path` mostrando progreso, salvo que el archivo ya exista.

    Si `dest_path` ya existe y force=False, no hace nada (usa la copia en caché).
    """
    if dest_path.exists() and not force:
        print(f"Ya existe en caché: {dest_path}")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, total_bytes)

        tmp_path.replace(dest_path)

    print()  # salto de línea tras la barra de progreso
    print(f"Descargado: {dest_path}")
    return dest_path


def _print_progress(downloaded: int, total: int) -> None:
    downloaded_mb = downloaded / (1024 * 1024)
    if total:
        pct = downloaded * 100 / total
        total_mb = total / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:5.1f}% ({downloaded_mb:,.1f} MB / {total_mb:,.1f} MB)")
    else:
        sys.stdout.write(f"\r  {downloaded_mb:,.1f} MB descargados")
    sys.stdout.flush()


def download_open_payments(force: bool = False):
    """Descarga (con caché) el CSV de CMS Open Payments PY2024 - General Payments."""
    dest_path = CMS_OPEN_PAYMENTS_RAW_DIR / OPEN_PAYMENTS_FILENAME
    return _download_with_cache(OPEN_PAYMENTS_URL, dest_path, force=force)


def download_leie(force: bool = False):
    """Descarga (con caché) el CSV de la base LEIE."""
    dest_path = LEIE_RAW_DIR / LEIE_FILENAME
    return _download_with_cache(LEIE_URL, dest_path, force=force)


def load_open_payments(force_download: bool = False) -> pd.DataFrame:
    """Descarga si hace falta y carga el dataset crudo de CMS Open Payments."""
    path = download_open_payments(force=force_download)
    return pd.read_csv(path, low_memory=False)


def load_leie(force_download: bool = False) -> pd.DataFrame:
    """Descarga si hace falta y carga la lista LEIE de proveedores excluidos."""
    path = download_leie(force=force_download)
    return pd.read_csv(path, low_memory=False)


if __name__ == "__main__":
    download_open_payments()
    download_leie()
