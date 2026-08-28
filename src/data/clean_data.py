"""Limpieza de datos y construcción de la etiqueta de fraude vía el cruce Open Payments x LEIE.

Estrategia de cruce en dos niveles (no se mezclan entre sí):

1. Etiqueta primaria (TARGET_COLUMN, columna `is_excluded`) - match estricto por NPI:
   se compara `Covered_Recipient_NPI` (Open Payments) contra `NPI` (LEIE), y solo
   cuenta como match si AMBOS lados tienen un NPI válido (no nulo, no cero). Esta es
   la etiqueta que se usa para entrenamiento supervisado.

   Limitación conocida: los registros de la LEIE sin NPI quedan fuera de este
   cruce estricto, así que el recall de `is_excluded` está limitado por la
   cobertura de NPI en la fuente (~11% de la LEIE trae NPI válido). Un
   proveedor excluido sin NPI registrado en la LEIE nunca podrá marcarse como
   `is_excluded=1` por esta vía.

2. Columna exploratoria secundaria - match difuso por nombre: para los registros
   de la LEIE SIN NPI válido, se intenta un match difuso contra Open Payments por
   apellido + nombre + estado (comparación insensible a mayúsculas, tolerante a
   pequeñas variaciones de escritura vía similitud de secuencias). El resultado se
   guarda en `is_excluded_name_match` (0/1) junto con `match_confidence` (score de
   similitud 0-1), completamente separado de `is_excluded` - nunca se combina con
   la etiqueta primaria ni se usa para entrenamiento supervisado.

Dado que Open Payments PY2024 pesa varios GB (~15.5M filas), se lee en chunks (no se
carga completo en memoria) y el resultado se escribe incrementalmente a data/processed/.

El output es una tabla liviana (Record_ID + columnas identificadoras + las 3 columnas
de etiqueta), NO una copia de las 91 columnas originales de Open Payments: escribir el
archivo completo (~9GB) en esta máquina (RAM limitada) tomaba varias horas. build_features.py
debe hacer merge de este resultado contra el CSV crudo por Record_ID para recuperar el
resto de columnas al construir features.
"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import (
    CMS_OPEN_PAYMENTS_RAW_DIR,
    LEIE_FILENAME,
    LEIE_RAW_DIR,
    OPEN_PAYMENTS_FILENAME,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)

# Similitud mínima (Dice sobre bigramas) que apellido Y nombre deben superar POR
# SEPARADO para siquiera considerarse candidatos. Sin este piso, un apellido común
# (p.ej. "REED") ya aporta 0.6*1.0=0.6 al score combinado aunque el nombre no se
# parezca en nada, inundando el match difuso de falsos positivos por apellido
# compartido. Ambos umbrales calibrados sobre pares con typos típicos (p.ej.
# SMITH/SMYTH ~0.50, ROBERT/ROBET ~0.67, JOHNSON/JOHNSTON ~0.77, MARIA/MARIE ~0.75).
#
# Esto es una decisión de modelado incorporada en la etiqueta `is_excluded_name_match`,
# no un default arbitrario: cambiar este valor cambia qué cuenta como match difuso, así
# que debe quedar visible para quien lea o retoque este código más adelante.
MIN_COMPONENT_SIMILARITY = 0.50
NAME_MATCH_THRESHOLD = 0.65

DEFAULT_CHUNKSIZE = 300_000
DEFAULT_OUTPUT_FILENAME = "fraud_labels.csv"

# Columnas mínimas necesarias de Open Payments: la clave para reunir con el resto de
# columnas más adelante (Record_ID) y las usadas en el cruce con la LEIE. Leer solo
# estas columnas (usecols) reduce drásticamente el tiempo de parseo del CSV de ~15.5M
# filas, comparado con parsear las 91 columnas originales.
OPEN_PAYMENTS_USECOLS = [
    "Record_ID",
    "Covered_Recipient_NPI",
    "Covered_Recipient_Type",
    "Covered_Recipient_First_Name",
    "Covered_Recipient_Last_Name",
    "Recipient_State",
]

_NON_ALPHA_SPACE_RE = re.compile(r"[^A-Z\s]")


def _normalize_name(value) -> str:
    """Normaliza un nombre para comparación: mayúsculas, sin puntuación, sin espacios extra."""
    if pd.isna(value):
        return ""
    text = str(value).upper().strip()
    text = _NON_ALPHA_SPACE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _bigrams(s: str) -> frozenset:
    """Set de bigramas de caracteres de `s`, usado como huella para similitud difusa rápida."""
    if len(s) < 2:
        return frozenset((s,)) if s else frozenset()
    return frozenset(s[i : i + 2] for i in range(len(s) - 1))


def _dice(a: frozenset, b: frozenset) -> float:
    """Coeficiente de Dice entre dos sets de bigramas: barato de calcular (solo
    intersección de sets) y suficientemente sensible a typos de 1-2 caracteres,
    a diferencia de difflib.SequenceMatcher que es demasiado lento a esta escala
    (~15M filas) por reconstruir su estado interno en cada comparación.
    """
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def clean_leie(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza NPI, nombre y estado de la LEIE para el cruce con Open Payments."""
    cleaned = df.copy()

    npi = cleaned["NPI"].astype(str).str.strip()
    invalid_npi = npi.isin(["", "0000000000"])
    cleaned["npi_norm"] = npi.where(~invalid_npi, "")

    cleaned["last_name_norm"] = cleaned["LASTNAME"].map(_normalize_name)
    cleaned["first_name_norm"] = cleaned["FIRSTNAME"].map(_normalize_name)
    cleaned["state_norm"] = cleaned["STATE"].astype(str).str.upper().str.strip()
    return cleaned


def clean_open_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza NPI, nombre y estado de un chunk de Open Payments para el cruce con la LEIE."""
    cleaned = df.copy()

    npi = cleaned["Covered_Recipient_NPI"]
    cleaned["_npi_valid"] = npi.notna() & (npi != 0)

    cleaned["_last_name_norm"] = cleaned["Covered_Recipient_Last_Name"].map(_normalize_name)
    cleaned["_first_name_norm"] = cleaned["Covered_Recipient_First_Name"].map(_normalize_name)
    cleaned["_state_norm"] = cleaned["Recipient_State"].astype(str).str.upper().str.strip()
    return cleaned


def _build_leie_lookups(leie_clean: pd.DataFrame):
    """Construye: (1) el set de NPIs excluidos válidos, (2) un índice por (estado, inicial de
    apellido) de los registros de la LEIE sin NPI, para el match difuso por nombre.
    """
    valid_npi_mask = leie_clean["npi_norm"] != ""
    valid_npi_set = set(leie_clean.loc[valid_npi_mask, "npi_norm"].astype("int64"))

    no_npi = leie_clean.loc[~valid_npi_mask]
    no_npi = no_npi[no_npi["last_name_norm"] != ""]

    # Bloqueo por (estado, primeras 2 letras del apellido): reduce drásticamente los
    # candidatos a comparar (imprescindible a la escala de Open Payments), a costa de
    # no detectar typos que caigan en esas 2 primeras letras del apellido.
    name_blocks: dict[tuple[str, str], list[tuple[frozenset, frozenset]]] = defaultdict(list)
    for state, last, first in zip(
        no_npi["state_norm"], no_npi["last_name_norm"], no_npi["first_name_norm"]
    ):
        name_blocks[(state, last[:2])].append((_bigrams(last), _bigrams(first)))
    return valid_npi_set, name_blocks


def _match_name_block(state: str, last: str, first: str, name_blocks: dict) -> tuple[int, float]:
    """Busca el mejor match difuso de `last`/`first` entre los candidatos del bloque
    (estado, 2 primeras letras del apellido), usando similitud de bigramas precomputados.
    """
    if not last:
        return 0, 0.0
    candidates = name_blocks.get((state, last[:2]), ())
    if not candidates:
        return 0, 0.0

    last_bg = _bigrams(last)
    first_bg = _bigrams(first)

    best_score = 0.0
    for cand_last_bg, cand_first_bg in candidates:
        last_ratio = _dice(last_bg, cand_last_bg)
        first_ratio = _dice(first_bg, cand_first_bg)
        if last_ratio < MIN_COMPONENT_SIMILARITY or first_ratio < MIN_COMPONENT_SIMILARITY:
            continue
        score = 0.6 * last_ratio + 0.4 * first_ratio
        if score > best_score:
            best_score = score
    return int(best_score >= NAME_MATCH_THRESHOLD), round(best_score, 4)


def label_fraud(
    open_payments_path: Path | None = None,
    leie_path: Path | None = None,
    output_path: Path | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Cruza Open Payments con la LEIE y escribe a data/processed/ una tabla de etiquetas
    (Record_ID + columnas identificadoras + is_excluded + is_excluded_name_match +
    match_confidence), lista para reunirse por Record_ID con el resto de columnas.

    Lee Open Payments en chunks de `chunksize` filas (el CSV pesa varios GB) y escribe
    el resultado incrementalmente, para no cargar el archivo completo en memoria. Devuelve
    un resumen con los conteos del cruce.
    """
    open_payments_path = open_payments_path or (CMS_OPEN_PAYMENTS_RAW_DIR / OPEN_PAYMENTS_FILENAME)
    leie_path = leie_path or (LEIE_RAW_DIR / LEIE_FILENAME)
    output_path = output_path or (PROCESSED_DATA_DIR / DEFAULT_OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    leie_raw = pd.read_csv(leie_path, dtype=str, keep_default_na=False)
    leie_clean = clean_leie(leie_raw)
    valid_npi_set, name_blocks = _build_leie_lookups(leie_clean)

    total_op_rows = 0
    primary_positive = 0
    secondary_positive = 0
    first_chunk = True

    # Cache de matches difusos por (estado, apellido, nombre) normalizados, persistente
    # entre chunks: el mismo destinatario suele recibir muchos pagos a lo largo de todo
    # el archivo, así que evita recalcular su similitud contra la LEIE más de una vez.
    match_cache: dict[tuple[str, str, str], tuple[int, float]] = {}

    reader = pd.read_csv(
        open_payments_path,
        chunksize=chunksize,
        usecols=OPEN_PAYMENTS_USECOLS,
        dtype={"Covered_Recipient_NPI": "Int64"},
        low_memory=False,
    )
    for chunk in reader:
        chunk = clean_open_payments(chunk)

        chunk[TARGET_COLUMN] = (
            chunk["_npi_valid"] & chunk["Covered_Recipient_NPI"].isin(valid_npi_set)
        ).astype(int)

        recipient_keys = list(
            zip(chunk["_state_norm"], chunk["_last_name_norm"], chunk["_first_name_norm"])
        )
        for key in set(recipient_keys):
            if key not in match_cache:
                match_cache[key] = _match_name_block(*key, name_blocks)
        chunk["is_excluded_name_match"] = [match_cache[k][0] for k in recipient_keys]
        chunk["match_confidence"] = [match_cache[k][1] for k in recipient_keys]

        chunk = chunk.drop(columns=["_npi_valid", "_last_name_norm", "_first_name_norm", "_state_norm"])

        chunk.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False

        total_op_rows += len(chunk)
        primary_positive += int(chunk[TARGET_COLUMN].sum())
        secondary_positive += int(chunk["is_excluded_name_match"].sum())

    return {
        "open_payments_rows": total_op_rows,
        "leie_rows": len(leie_raw),
        "primary_npi_match_positive": primary_positive,
        "secondary_name_match_positive": secondary_positive,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    summary = label_fraud()
    for key, value in summary.items():
        print(f"{key}: {value}")
