"""Tabla a nivel PROVEEDOR (una fila por NPI) para modelar is_excluded en su unidad real.

is_excluded es un dato por proveedor (el cruce estricto con la LEIE es por NPI en
clean_data.py), pero todo el modelado previo era por pago, y eso dejó que el modelo
memorizara proveedores excluidos concretos (ver diagnose_provider_generalization.py). Este
módulo agrega fraud_features.csv a una fila por proveedor.

## Recuperación del NPI

fraud_features.csv no trae NPI; fraud_labels.csv sí. `train_supervised.attach_provider_ids`
sirve para una muestra (~391k filas): arma el índice de Record_ID de la muestra en memoria y
recorre fraud_labels.csv completo por llamada. Aplicado a las 15.5M filas exigiría ~0.9GB de
pico (o 52 recorridos completos si se llamara por chunk), así que acá se leen ambos CSVs en
paralelo, chunk a chunk: se generaron en el mismo orden y nunca se reordenan (mismo patrón que
build_dashboard_summary.py), y cada par de chunks se verifica por Record_ID - si no coinciden,
falla en vez de cruzar filas mal.

Las filas SIN NPI válido (hospitales docentes, ~0.3%, facturan como institución y siempre
tienen is_excluded=0) se excluyen por completo: no son un proveedor individual.

## Por qué en dos pasadas (partición en disco)

Los pagos de un proveedor están dispersos por todo el archivo, y la mediana por proveedor no
se puede acumular chunk a chunk. Primero se reparte cada pago (columnas mínimas, strings
codificados como enteros) en N_BUCKETS archivos parquet temporales según NPI % N_BUCKETS;
después se agrega cada bucket entero en memoria (~1M filas) con groupbys exactos. Ningún
proveedor queda partido entre buckets.

## Salidas (en data/processed/, ya gitignoreado)

- provider_features.csv: una fila por proveedor - is_excluded, conteo y monto total,
  media/mediana de payment_amount_log, especialidad, concentración de fabricantes (cantidad
  de fabricantes distintos y share del monto del fabricante principal), mezcla de
  payment_nature y share de pagos con is_related_product / is_third_party_payment.
- provider_manufacturer_amounts.parquet: (npi, fabricante, cantidad de pagos, monto). El
  riesgo promedio ponderado por fabricante NO se precalcula acá: es un target encoding y
  debe ajustarse solo sobre los proveedores de train, que recién existen tras el split de
  train_provider_level.py. Esta tabla es lo que ese paso necesita para calcularlo sin fuga.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import PROCESSED_DATA_DIR, TARGET_COLUMN
from src.data.clean_data import DEFAULT_OUTPUT_FILENAME as FRAUD_LABELS_FILENAME
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME

DEFAULT_CHUNKSIZE = 300_000
N_BUCKETS = 16
PROVIDER_FEATURES_FILENAME = "provider_features.csv"
PROVIDER_MANUFACTURER_FILENAME = "provider_manufacturer_amounts.parquet"
NPI_COLUMN = "Covered_Recipient_NPI"

# payment_nature con share explícito: los de mayor señal en la EDA (Debt forgiveness ~19x la
# tasa base, Consulting Fee ~9.6x, Long term medical supply or device loan ~3.9x) más las de
# mayor volumen (Food and Beverage es el 91% de los pagos; Travel and Lodging, Education y
# Compensation for services son las siguientes). El resto de categorías (Gift, Honoraria,
# Grant, Royalty, etc.: chicas y/o sin exclusiones en la EDA) se agrupa en "other".
NATURE_GROUPS = {
    "Food and Beverage": "food_beverage",
    "Consulting Fee": "consulting_fee",
    "Debt forgiveness": "debt_forgiveness",
    "Long term medical supply or device loan": "device_loan",
    "Travel and Lodging": "travel_lodging",
    "Education": "education",
    "Compensation for services other than consulting, including serving as faculty or as a "
    "speaker at a venue other than a continuing education program": "speaker_compensation",
}
NATURE_GROUP_NAMES = list(NATURE_GROUPS.values()) + ["other"]

FEATURES_USECOLS = [
    "Record_ID", "payment_amount", "payment_amount_log", "payment_nature", "recipient_specialty",
    "manufacturer_name", "is_related_product", "is_third_party_payment", TARGET_COLUMN,
]
SPILL_SCHEMA = pa.schema([
    ("npi", pa.int64()),
    ("recipient_specialty", pa.int32()),
    ("manufacturer", pa.int32()),
    ("nature_group", pa.int8()),
    ("payment_amount", pa.float64()),
    ("payment_amount_log", pa.float32()),
    ("is_related_product", pa.int8()),
    ("is_third_party_payment", pa.int8()),
    (TARGET_COLUMN, pa.int8()),
])


def _to_global_ids(values: pd.Series, vocab: dict) -> np.ndarray:
    """Codifica strings como enteros estables entre chunks (vocab crece a medida que aparecen
    valores nuevos). Nulo -> "Missing", igual que train_supervised.py antes de codificar."""
    codes, uniques = pd.factorize(values.fillna("Missing"))
    lookup = np.array([vocab.setdefault(u, len(vocab)) for u in uniques], dtype=np.int32)
    return lookup[codes]


def _nature_group_codes(nature: pd.Series) -> np.ndarray:
    slugs = nature.map(NATURE_GROUPS).fillna("other")
    return pd.Categorical(slugs, categories=NATURE_GROUP_NAMES).codes.astype(np.int8)


def aggregate_provider_rows(payments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Agrega pagos (una fila por pago, todos con NPI válido y TODOS los pagos de cada NPI
    presente) a una fila por proveedor. Columnas esperadas: npi, recipient_specialty,
    manufacturer, nature_group (nombres de NATURE_GROUP_NAMES), payment_amount,
    payment_amount_log, is_related_product, is_third_party_payment, is_excluded.

    Devuelve (providers, provider_manufacturer, stats). Falla si un proveedor tiene pagos
    con is_excluded distinto (debería ser constante: es un dato por NPI)."""
    g = payments.groupby("npi")
    providers = pd.DataFrame({
        TARGET_COLUMN: g[TARGET_COLUMN].max(),
        "_excluded_min": g[TARGET_COLUMN].min(),
        "total_payment_count": g.size(),
        "total_payment_amount": g["payment_amount"].sum(),
        "mean_payment_amount_log": g["payment_amount_log"].mean(),
        "median_payment_amount_log": g["payment_amount_log"].median(),
        "share_related_product": g["is_related_product"].mean(),
        "share_third_party_payment": g["is_third_party_payment"].mean(),
    })
    inconsistent = int((providers[TARGET_COLUMN] != providers.pop("_excluded_min")).sum())
    if inconsistent:
        raise RuntimeError(f"{inconsistent} proveedores tienen pagos con is_excluded distinto - no es constante por NPI.")

    # Especialidad: debería ser fija por proveedor; si no, se usa la moda (desempate estable).
    spec = payments.groupby(["npi", "recipient_specialty"]).size().rename("n").reset_index()
    n_specialties = spec.groupby("npi").size()
    mode = spec.sort_values(["npi", "n", "recipient_specialty"], ascending=[True, False, True]).drop_duplicates("npi")
    providers["recipient_specialty"] = mode.set_index("npi")["recipient_specialty"]

    # Concentración de fabricantes, desde la tabla (npi, fabricante).
    pairs = (
        payments.groupby(["npi", "manufacturer"])
        .agg(payment_count=("payment_amount", "size"), amount=("payment_amount", "sum"))
        .reset_index()
    )
    by_npi = pairs.groupby("npi")
    providers["n_distinct_manufacturers"] = by_npi.size()
    providers["top_manufacturer_amount_share"] = by_npi["amount"].max() / providers["total_payment_amount"]

    # Mezcla de payment_nature: share de los PAGOS del proveedor en cada grupo.
    mix = pd.crosstab(payments["npi"], payments["nature_group"], normalize="index")
    mix = mix.reindex(columns=NATURE_GROUP_NAMES, fill_value=0.0).add_prefix("share_nature_")
    providers = providers.join(mix)

    stats = {"providers": len(providers), "providers_with_multiple_specialties": int((n_specialties > 1).sum())}
    return providers.reset_index(), pairs, stats


def _spill_to_buckets(features_path: Path, labels_path: Path, tmp_dir: Path, chunksize: int, n_buckets: int):
    vocabs = {"recipient_specialty": {}, "manufacturer": {}}
    stats = {"rows_total": 0, "rows_without_npi": 0, "excluded_rows_without_npi": 0}
    writers = [pq.ParquetWriter(tmp_dir / f"bucket_{b:02d}.parquet", SPILL_SCHEMA) for b in range(n_buckets)]
    try:
        features_reader = pd.read_csv(features_path, usecols=FEATURES_USECOLS, chunksize=chunksize, low_memory=False)
        labels_reader = pd.read_csv(
            labels_path, usecols=["Record_ID", NPI_COLUMN], dtype={NPI_COLUMN: "Int64"}, chunksize=chunksize
        )
        for features, labels in zip(features_reader, labels_reader, strict=True):
            if len(features) != len(labels) or not np.array_equal(
                features["Record_ID"].to_numpy(), labels["Record_ID"].to_numpy()
            ):
                raise RuntimeError(
                    "fraud_features.csv y fraud_labels.csv no están alineados fila a fila en este chunk - "
                    "no se puede asignar el NPI por posición."
                )
            npi = labels[NPI_COLUMN]
            valid = (npi.notna() & (npi != 0)).to_numpy()
            stats["rows_total"] += len(features)
            stats["rows_without_npi"] += int((~valid).sum())
            stats["excluded_rows_without_npi"] += int(features.loc[~valid, TARGET_COLUMN].sum())

            f = features[valid]
            npi_values = npi[valid].astype("int64").to_numpy()
            spill = pd.DataFrame({
                "npi": npi_values,
                "recipient_specialty": _to_global_ids(f["recipient_specialty"], vocabs["recipient_specialty"]),
                "manufacturer": _to_global_ids(f["manufacturer_name"], vocabs["manufacturer"]),
                "nature_group": _nature_group_codes(f["payment_nature"]),
                "payment_amount": f["payment_amount"].to_numpy("float64"),
                "payment_amount_log": f["payment_amount_log"].to_numpy("float32"),
                "is_related_product": f["is_related_product"].to_numpy("int8"),
                "is_third_party_payment": f["is_third_party_payment"].to_numpy("int8"),
                TARGET_COLUMN: f[TARGET_COLUMN].to_numpy("int8"),
            })
            buckets = npi_values % n_buckets
            for b in range(n_buckets):
                part = spill[buckets == b]
                if len(part):
                    writers[b].write_table(pa.Table.from_pandas(part, schema=SPILL_SCHEMA, preserve_index=False))
    finally:
        for w in writers:
            w.close()
    return vocabs, stats


def build_provider_table(
    features_path: Path | None = None,
    labels_path: Path | None = None,
    output_path: Path | None = None,
    pairs_path: Path | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    n_buckets: int = N_BUCKETS,
) -> dict:
    features_path = features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME)
    labels_path = labels_path or (PROCESSED_DATA_DIR / FRAUD_LABELS_FILENAME)
    output_path = output_path or (PROCESSED_DATA_DIR / PROVIDER_FEATURES_FILENAME)
    pairs_path = pairs_path or (PROCESSED_DATA_DIR / PROVIDER_MANUFACTURER_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        vocabs, stats = _spill_to_buckets(features_path, labels_path, Path(tmp), chunksize, n_buckets)
        print(f"[{time.time() - start:6.1f}s] pass 1 done: {stats['rows_total']:,} rows partitioned into {n_buckets} buckets")

        specialty_names = np.array(list(vocabs["recipient_specialty"]), dtype=object)
        manufacturer_names = np.array(list(vocabs["manufacturer"]), dtype=object)
        stats.update(providers=0, excluded_providers=0, providers_with_multiple_specialties=0, provider_manufacturer_pairs=0)
        pairs_writer = None
        try:
            for b in range(n_buckets):
                payments = pq.read_table(Path(tmp) / f"bucket_{b:02d}.parquet").to_pandas()
                payments["nature_group"] = pd.Categorical.from_codes(payments["nature_group"], categories=NATURE_GROUP_NAMES)
                providers, pairs, bucket_stats = aggregate_provider_rows(payments)
                del payments

                providers["recipient_specialty"] = specialty_names[providers["recipient_specialty"].to_numpy()]
                providers.to_csv(output_path, mode="w" if b == 0 else "a", header=b == 0, index=False)

                pairs_out = pa.table({
                    "npi": pairs["npi"].to_numpy("int64"),
                    "manufacturer_name": pa.array(manufacturer_names[pairs["manufacturer"].to_numpy()]).dictionary_encode(),
                    "payment_count": pairs["payment_count"].to_numpy("int32"),
                    "amount": pairs["amount"].to_numpy("float64"),
                })
                if pairs_writer is None:
                    pairs_writer = pq.ParquetWriter(pairs_path, pairs_out.schema)
                pairs_writer.write_table(pairs_out)

                stats["providers"] += bucket_stats["providers"]
                stats["excluded_providers"] += int(providers[TARGET_COLUMN].sum())
                stats["providers_with_multiple_specialties"] += bucket_stats["providers_with_multiple_specialties"]
                stats["provider_manufacturer_pairs"] += len(pairs)
        finally:
            if pairs_writer is not None:
                pairs_writer.close()

    stats["runtime_seconds"] = time.time() - start
    stats["output_path"] = str(output_path)
    stats["pairs_path"] = str(pairs_path)
    return stats


if __name__ == "__main__":
    summary = build_provider_table()
    for key, value in summary.items():
        print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
