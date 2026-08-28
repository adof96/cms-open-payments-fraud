"""Borra el contenido descargado en data/raw/ (CMS Open Payments y LEIE) para liberar espacio.

Uso:
    python -m src.data.clean_data_cache
"""

from src.config import CMS_OPEN_PAYMENTS_RAW_DIR, LEIE_RAW_DIR, RAW_DATA_DIR


def clean_raw_data_cache() -> None:
    """Elimina todos los archivos descargados en las subcarpetas de data/raw/, sin borrar las carpetas."""
    for raw_dir in (CMS_OPEN_PAYMENTS_RAW_DIR, LEIE_RAW_DIR):
        if not raw_dir.exists():
            continue
        for path in raw_dir.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_file():
                path.unlink()
        print(f"Limpiado: {raw_dir}")


if __name__ == "__main__":
    print(f"Borrando caché de datos crudos en {RAW_DATA_DIR} ...")
    clean_raw_data_cache()
