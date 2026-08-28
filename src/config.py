"""Configuración central del proyecto: rutas y constantes compartidas."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Subcarpetas de data/raw/ por fuente
CMS_OPEN_PAYMENTS_RAW_DIR = RAW_DATA_DIR / "cms_open_payments"
LEIE_RAW_DIR = RAW_DATA_DIR / "leie"

# CMS Open Payments — Program Year 2024, dataset "General Payments" (detallado).
# https://openpaymentsdata.cms.gov/dataset/e6b17c6a-2534-4207-a4a1-6746a14911ff
OPEN_PAYMENTS_URL = (
    "https://download.cms.gov/openpayments/PGYR2024_P06302026_06032026/"
    "OP_DTL_GNRL_PGYR2024_P06302026_06032026.csv"
)
OPEN_PAYMENTS_FILENAME = "OP_DTL_GNRL_PGYR2024.csv"

# LEIE (List of Excluded Individuals/Entities) — base completa, actualizada mensualmente.
# https://oig.hhs.gov/exclusions/exclusions_list.asp
LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
LEIE_FILENAME = "leie.csv"

# Nombre de la columna objetivo derivada del cruce Open Payments x LEIE
TARGET_COLUMN = "is_excluded"

RANDOM_STATE = 42
