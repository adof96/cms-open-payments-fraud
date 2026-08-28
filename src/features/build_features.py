"""Construcción de la tabla de features para el modelo de detección de fraude.

Fusiona data/processed/fraud_labels.csv (las etiquetas, producidas por
src/data/clean_data.py) con el CSV crudo de CMS Open Payments por `Record_ID`, para
recuperar las columnas necesarias para features. fraud_labels.csv es una tabla liviana
(Record_ID + columnas identificadoras + is_excluded + is_excluded_name_match +
match_confidence) y NO trae las 91 columnas originales de Open Payments: escribirlas
todas para ~15.5M filas en esta máquina (RAM limitada) tomaba horas (ver docstring de
clean_data.py). Por eso este módulo vuelve a leer el CSV crudo para el resto de columnas.

Columnas leídas del crudo (además de `Record_ID`, la clave del merge) y por qué:
- Covered_Recipient_NPI: identificador del destinatario; se usa para la feature de
  frecuencia de pagos por destinatario (`payment_frequency`).
- Covered_Recipient_Specialty_1: especialidad del proveedor (feature categórica; los
  patrones de pago varían mucho por especialidad).
- Total_Amount_of_Payment_USDollars: monto del pago, la feature numérica más directa
  para anomalías (montos atípicamente altos/bajos).
- Number_of_Payments_Included_in_Total_Amount: cuántos pagos individuales agrega la fila.
- Form_of_Payment_or_Transfer_of_Value / Nature_of_Payment_or_Transfer_of_Value: forma
  (efectivo, en especie, acciones...) y naturaleza (comida, consultoría, viaje...) del
  pago - categorías con distinta propensión a fraude.
- Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name / _ID: empresa pagadora,
  para features agregadas por fabricante.
- Date_of_Payment: fecha del pago, para features de estacionalidad (mes, día de semana).
- Physician_Ownership_Indicator: si el destinatario tiene participación de propiedad en
  la empresa pagadora - señal directa de conflicto de interés.
- Third_Party_Payment_Recipient_Indicator: si el pago pasó por un tercero (Entity/
  Individual) en vez de ir directo al destinatario.
- Charity_Indicator: si el pago fue a una entidad benéfica.
- Related_Product_Indicator: si el pago está asociado a un producto/droga/dispositivo.
- Dispute_Status_for_Publication: si el destinatario disputó la publicación del pago.

Igual que en clean_data.py, el CSV crudo (~8.5GB, ~15.5M filas) se lee en chunks - no se
carga completo en memoria - dado que esta máquina tiene RAM libre limitada (~1.3GB).

`payment_frequency` (pagos totales por NPI) requiere ver el archivo completo, porque los
pagos de un mismo destinatario están dispersos por todo el archivo, no agrupados en un
mismo chunk. Se calcula en una pasada separada y liviana sobre fraud_labels.csv (que ya
trae Covered_Recipient_NPI), no sobre el crudo de 8.5GB.

Balance de clases: `is_excluded` (la etiqueta primaria) tiene ~3,809 positivos sobre
~15,498,687 filas - aproximadamente 1:4,070 positivo:negativo. Este módulo NO aplica
ningún resampling ni ponderación de clases: la estrategia para manejar ese desbalance
(class_weight, undersampling, SMOTE, etc.) es una decisión del paso de modelado
(train_supervised.py / train_unsupervised.py), no de la construcción de features.
`is_excluded`, `is_excluded_name_match` y `match_confidence` se mantienen como columnas
separadas en el output - cuál se usa como TARGET_COLUMN se decide en el modelado.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CMS_OPEN_PAYMENTS_RAW_DIR, OPEN_PAYMENTS_FILENAME, PROCESSED_DATA_DIR, TARGET_COLUMN
from src.data.clean_data import DEFAULT_OUTPUT_FILENAME as FRAUD_LABELS_FILENAME

DEFAULT_CHUNKSIZE = 300_000
DEFAULT_OUTPUT_FILENAME = "fraud_features.csv"

LABEL_USECOLS = ["Record_ID", TARGET_COLUMN, "is_excluded_name_match", "match_confidence"]

RAW_FEATURE_USECOLS = [
    "Record_ID",
    "Covered_Recipient_NPI",
    "Covered_Recipient_Specialty_1",
    "Total_Amount_of_Payment_USDollars",
    "Number_of_Payments_Included_in_Total_Amount",
    "Form_of_Payment_or_Transfer_of_Value",
    "Nature_of_Payment_or_Transfer_of_Value",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID",
    "Date_of_Payment",
    "Physician_Ownership_Indicator",
    "Third_Party_Payment_Recipient_Indicator",
    "Charity_Indicator",
    "Related_Product_Indicator",
    "Dispute_Status_for_Publication",
]


def _compute_payment_frequency(labels_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> dict:
    """Cuenta cuántas filas de pago tiene cada Covered_Recipient_NPI en todo el archivo.

    Se calcula en una pasada separada sobre fraud_labels.csv (liviana, ya trae la
    columna) en vez de sobre el CSV crudo de 8.5GB: hace falta ver el archivo completo
    para contar bien, ya que los pagos de un mismo destinatario están dispersos por
    todo el archivo, no agrupados en un mismo chunk.
    """
    counts: Counter = Counter()
    reader = pd.read_csv(
        labels_path,
        usecols=["Covered_Recipient_NPI"],
        dtype={"Covered_Recipient_NPI": "Int64"},
        chunksize=chunksize,
    )
    for chunk in reader:
        counts.update(chunk["Covered_Recipient_NPI"].dropna().astype("int64").tolist())
    return dict(counts)


def build_features(raw_chunk: pd.DataFrame, payment_frequency: dict) -> pd.DataFrame:
    """Construye las features de un chunk de Open Payments (ya con las columnas de
    RAW_FEATURE_USECOLS) a partir de sus columnas crudas.
    """
    features = pd.DataFrame(index=raw_chunk.index)
    features["Record_ID"] = raw_chunk["Record_ID"]

    amount = raw_chunk["Total_Amount_of_Payment_USDollars"]
    features["payment_amount"] = amount
    features["payment_amount_log"] = np.log1p(amount.clip(lower=0))

    features["num_payments_included"] = raw_chunk["Number_of_Payments_Included_in_Total_Amount"]
    features["payment_form"] = raw_chunk["Form_of_Payment_or_Transfer_of_Value"]
    features["payment_nature"] = raw_chunk["Nature_of_Payment_or_Transfer_of_Value"]
    features["recipient_specialty"] = raw_chunk["Covered_Recipient_Specialty_1"]
    features["manufacturer_name"] = raw_chunk["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"]
    features["manufacturer_id"] = raw_chunk["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID"]

    payment_date = pd.to_datetime(raw_chunk["Date_of_Payment"], format="%m/%d/%Y", errors="coerce")
    features["payment_month"] = payment_date.dt.month
    features["payment_day_of_week"] = payment_date.dt.dayofweek

    features["is_ownership_interest"] = (raw_chunk["Physician_Ownership_Indicator"] == "Yes").astype(int)
    features["is_third_party_payment"] = (
        raw_chunk["Third_Party_Payment_Recipient_Indicator"].ne("No Third Party Payment").astype(int)
    )
    features["is_charity"] = (raw_chunk["Charity_Indicator"] == "Yes").astype(int)
    features["is_related_product"] = (raw_chunk["Related_Product_Indicator"] == "Yes").astype(int)
    features["is_disputed"] = (raw_chunk["Dispute_Status_for_Publication"] == "Yes").astype(int)

    features["payment_frequency"] = raw_chunk["Covered_Recipient_NPI"].map(payment_frequency)

    return features


def build_feature_table(
    open_payments_path: Path | None = None,
    labels_path: Path | None = None,
    output_path: Path | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    """Fusiona fraud_labels.csv con el CSV crudo de Open Payments por Record_ID y escribe
    la tabla de features a data/processed/. Lee ambos archivos en chunks para no cargar
    el crudo completo en memoria. Devuelve un resumen con conteos del resultado.
    """
    open_payments_path = open_payments_path or (CMS_OPEN_PAYMENTS_RAW_DIR / OPEN_PAYMENTS_FILENAME)
    labels_path = labels_path or (PROCESSED_DATA_DIR / FRAUD_LABELS_FILENAME)
    output_path = output_path or (PROCESSED_DATA_DIR / DEFAULT_OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payment_frequency = _compute_payment_frequency(labels_path, chunksize=chunksize)

    # Solo las columnas de etiqueta (no las identificadoras que fraud_labels.csv también
    # trae) para mantener el merge liviano: las identificadoras se vuelven a leer del
    # crudo junto con el resto de features.
    labels = pd.read_csv(
        labels_path,
        usecols=LABEL_USECOLS,
        dtype={"Record_ID": "int64", TARGET_COLUMN: "int8", "is_excluded_name_match": "int8"},
    )

    total_rows = 0
    first_chunk = True

    reader = pd.read_csv(
        open_payments_path,
        chunksize=chunksize,
        usecols=RAW_FEATURE_USECOLS,
        dtype={"Record_ID": "int64", "Covered_Recipient_NPI": "Int64"},
        low_memory=False,
    )
    for raw_chunk in reader:
        features = build_features(raw_chunk, payment_frequency)
        merged = features.merge(labels, on="Record_ID", how="left")

        merged.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        total_rows += len(merged)

    return {
        "output_rows": total_rows,
        "feature_columns": len(merged.columns) if total_rows else 0,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    summary = build_feature_table()
    for key, value in summary.items():
        print(f"{key}: {value}")
