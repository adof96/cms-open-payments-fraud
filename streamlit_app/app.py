"""App de Streamlit para explorar predicciones de fraude sobre CMS Open Payments.

Dos tabs:

- **Explore Results**: dashboard de solo lectura. Carga ÚNICAMENTE
  data/processed/dashboard_summary.json (precalculado por
  src/inference/build_dashboard_summary.py) - esta app NUNCA lee fraud_features.csv
  (3.39GB) ni xgboost_predictions.csv (414MB) directamente.
- **Try It Yourself**: scoring en vivo de un pago hipotético. Carga
  models/xgboost_is_excluded.pkl y models/supervised_preprocessing.pkl (vía
  model_io.py) y aplica el preprocesamiento con
  src.models.train_supervised.apply_preprocessing - la MISMA función que usan
  predictor.py y tune_threshold.py, no una reimplementación. No se reentrena ni se
  reajusta nada acá.

## Nota de calibración

La probabilidad que devuelve XGBoost en esta app es un **score de riesgo relativo**,
no una probabilidad real calibrada de que un pago sea fraudulento: el modelo se
entrenó con `scale_pos_weight` sobre una muestra con ~1:100 negativos:positivos (ver
train_supervised.py), muy distinto al ~1:4,070 real de la población completa. Eso
mejora la capacidad del modelo para separar casos de riesgo alto/bajo, pero significa
que sus probabilidades de salida están sistemáticamente infladas respecto a la tasa
real de exclusión - útil para *rankear* pagos por riesgo, no para leer "hay un X% de
probabilidad real de que esto sea fraude".
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit ejecuta este archivo con streamlit_app/ como directorio de trabajo, no la
# raíz del proyecto - sin esto, `from src...` falla con ModuleNotFoundError al correr
# `streamlit run streamlit_app/app.py` (no ocurre bajo AppTest/pytest, que sí corren
# desde la raíz).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.inference.build_dashboard_summary import DEFAULT_OUTPUT_FILENAME as SUMMARY_FILENAME
from src.models.model_io import load_model
from src.models.train_supervised import apply_preprocessing
from src.inference.predictor import THRESHOLD

SUMMARY_PATH = PROCESSED_DATA_DIR / SUMMARY_FILENAME

# Categorías de payment_nature de EDA sección 8 con tasa de exclusión muy por encima
# del baseline (~19x y ~9.6x respectivamente) - se resaltan en el gráfico de barras.
HIGH_RISK_PAYMENT_NATURE = {"Debt forgiveness", "Consulting Fee"}

# Defaults para las columnas del form simplificado de "Try It Yourself" que NO vienen
# de supervised_preprocessing.pkl (ese artefacto solo guarda payment_frequency_median
# - no guarda moda/mediana de payment_month, payment_day_of_week ni
# num_payments_included). Son aproximaciones razonables, no valores medidos del fold
# de entrenamiento:
DEFAULT_NUM_PAYMENTS_INCLUDED = 1  # la inmensa mayoría de las filas trae exactamente 1
DEFAULT_PAYMENT_MONTH = 6  # mitad de año, sin estacionalidad fuerte conocida
DEFAULT_PAYMENT_DAY_OF_WEEK = 2  # miércoles, día "neutro" entre semana


@st.cache_data
def load_summary() -> dict:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model_and_artifacts():
    model = load_model(MODELS_DIR / "xgboost_is_excluded.pkl")
    artifacts = load_model(MODELS_DIR / "supervised_preprocessing.pkl")
    return model, artifacts


def _category_bar_chart(category_stats: dict, title: str, highlight: set[str] | None = None):
    names = list(category_stats.keys())
    rates = [100 * v["flagged"] / v["count"] for v in category_stats.values()]
    order = np.argsort(rates)
    names = [names[i] for i in order]
    rates = [rates[i] for i in order]
    colors = ["#d62728" if (highlight and n in highlight) else "#4c72b0" for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
    ax.barh([n[:50] for n in names], rates, color=colors)
    ax.set_xlabel("Flagged rate (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def render_explore_tab(summary: dict) -> None:
    overall = summary["overall"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total payments scored", f"{overall['total_rows']:,}")
    col2.metric(
        "Flagged for review",
        f"{overall['total_flagged']:,}",
        f"{overall['flagged_rate'] * 100:.2f}% of all payments",
    )
    col3.metric(
        "Known exclusions (is_excluded)",
        f"{overall['total_excluded']:,}",
        f"{overall['excluded_rate'] * 100:.3f}% of all payments",
    )

    st.caption(
        "Threshold used for `flagged_for_review`: "
        f"**{THRESHOLD}** (chosen in tune_threshold.py for recall≈0.80 on the held-out test set)."
    )

    st.subheader("Flagged rate by category")
    tab_specialty, tab_manufacturer, tab_nature = st.tabs(
        ["Recipient specialty", "Manufacturer", "Payment nature"]
    )
    with tab_specialty:
        st.pyplot(
            _category_bar_chart(
                summary["category_flagged_rate"]["recipient_specialty"],
                "Flagged rate — top 20 recipient specialties by volume",
            )
        )
    with tab_manufacturer:
        st.pyplot(
            _category_bar_chart(
                summary["category_flagged_rate"]["manufacturer_name"],
                "Flagged rate — top 20 manufacturers by volume",
            )
        )
    with tab_nature:
        st.pyplot(
            _category_bar_chart(
                summary["category_flagged_rate"]["payment_nature"],
                "Flagged rate — all payment_nature categories",
                highlight=HIGH_RISK_PAYMENT_NATURE,
            )
        )
        st.caption(
            "Red bars: **Debt forgiveness** and **Consulting Fee** — flagged as high-risk "
            "in the EDA (~19x and ~9.6x the overall exclusion rate)."
        )

    st.subheader("Payment amount distribution (log scale)")
    hist = summary["amount_log_histogram"]
    edges = np.array(hist["bin_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(centers, hist["not_flagged"], width=width, alpha=0.6, label="Not flagged")
    ax.bar(centers, hist["flagged"], width=width, alpha=0.6, label="Flagged for review")
    ax.set_xlabel("log1p(payment amount, USD)")
    ax.set_ylabel("Row count")
    ax.set_yscale("log")
    ax.set_title("payment_amount_log distribution — flagged vs. not flagged")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Precision / recall tradeoff — XGBoost")
    sweep = pd.DataFrame(summary["threshold_sweep"]["xgboost"])
    threshold = st.slider(
        "Decision threshold", min_value=0.0, max_value=1.0, value=THRESHOLD, step=0.01,
        help="Looks up precomputed precision/recall/F1 at this threshold - no live model inference.",
    )
    row = sweep.iloc[(sweep["threshold"] - threshold).abs().idxmin()]

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{row['precision']:.3f}")
    m2.metric("Recall", f"{row['recall']:.3f}")
    m3.metric("F1", f"{row['f1']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sweep["recall"], sweep["precision"], color="#4c72b0")
    ax.scatter([row["recall"]], [row["precision"]], color="#d62728", zorder=5, label=f"threshold={row['threshold']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall curve — XGBoost")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def _build_input_row(
    payment_amount: float,
    payment_nature: str,
    payment_form: str,
    recipient_specialty: str,
    manufacturer_name: str,
    is_related_product: bool,
    is_third_party_payment: bool,
    payment_frequency_default: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "payment_amount_log": np.log1p(max(payment_amount, 0.0)),
                "num_payments_included": DEFAULT_NUM_PAYMENTS_INCLUDED,
                "payment_month": DEFAULT_PAYMENT_MONTH,
                "payment_day_of_week": DEFAULT_PAYMENT_DAY_OF_WEEK,
                "payment_frequency": payment_frequency_default,
                "is_related_product": int(is_related_product),
                "is_third_party_payment": int(is_third_party_payment),
                "recipient_specialty": recipient_specialty,
                "manufacturer_name": manufacturer_name,
                "payment_form": payment_form,
                "payment_nature": payment_nature,
            }
        ]
    )


def render_try_it_tab() -> None:
    model, artifacts = load_model_and_artifacts()

    payment_nature_options = sorted(artifacts["bucket_categories"]["payment_nature"]) + ["Other"]
    payment_form_options = sorted(artifacts["bucket_categories"]["payment_form"]) + ["Other"]
    specialty_options = sorted(artifacts["target_encoding"]["recipient_specialty"]["mapping"].keys())
    manufacturer_options = sorted(artifacts["target_encoding"]["manufacturer_name"]["mapping"].keys())

    st.write(
        "Fill in a hypothetical payment below. Fields not shown here use typical defaults: "
        f"{DEFAULT_NUM_PAYMENTS_INCLUDED} payment per record, month {DEFAULT_PAYMENT_MONTH} (June), "
        f"day of week {DEFAULT_PAYMENT_DAY_OF_WEEK} (Wednesday), and the recipient's historical "
        f"payment frequency set to the training median ({artifacts['payment_frequency_median']:.0f})."
    )

    with st.form("try_it_form"):
        payment_amount = st.number_input("Payment amount (USD)", min_value=0.0, value=100.0, step=10.0)
        payment_nature = st.selectbox("Payment nature", payment_nature_options)
        payment_form = st.selectbox("Payment form", payment_form_options)
        recipient_specialty = st.selectbox("Recipient specialty", specialty_options)
        manufacturer_name = st.selectbox("Manufacturer", manufacturer_options)
        is_related_product = st.toggle("Payment tied to a specific drug/device/biological?", value=True)
        is_third_party_payment = st.toggle("Paid through a third party (not directly to the recipient)?", value=False)
        submitted = st.form_submit_button("Score this payment")

    if not submitted:
        return

    raw_row = _build_input_row(
        payment_amount=payment_amount,
        payment_nature=payment_nature,
        payment_form=payment_form,
        recipient_specialty=recipient_specialty,
        manufacturer_name=manufacturer_name,
        is_related_product=is_related_product,
        is_third_party_payment=is_third_party_payment,
        payment_frequency_default=artifacts["payment_frequency_median"],
    )
    X = apply_preprocessing(raw_row, artifacts)
    probability = float(model.predict_proba(X)[:, 1][0])
    flagged = probability >= THRESHOLD

    st.divider()
    st.subheader("Result")
    st.progress(min(probability, 1.0))
    # 4 decimales (no 2-3): la tasa base de is_excluded es ~0.025%, así que la mayoría
    # de los pagos "típicos" da un score bajo pero informativo (p.ej. 0.0044) que un
    # formato .3f redondearía a un "0.000" engañoso.
    if flagged:
        st.error(f"**Flagged for review** — risk score {probability:.4f} ({probability:.2%}), threshold {THRESHOLD}")
    else:
        st.success(f"**Not flagged** — risk score {probability:.4f} ({probability:.2%}), threshold {THRESHOLD}")
    st.caption(
        "This is a **relative risk score**, not a calibrated real-world probability of fraud. "
        "The model was trained on a sample with roughly 1 exclusion per 100 payments (with class "
        "weighting), far richer in exclusions than the real population (~1 in 4,070). That helps it "
        "rank payments by risk, but its scores are systematically inflated relative to the true "
        "exclusion rate — use it to compare payments, not to read off an actual chance of fraud."
    )


def main() -> None:
    st.set_page_config(page_title="CMS Open Payments Fraud Detection", layout="wide")
    st.title("CMS Open Payments Fraud Detection")

    tab_explore, tab_try_it = st.tabs(["Explore Results", "Try It Yourself"])
    with tab_explore:
        if not SUMMARY_PATH.exists():
            st.error(
                f"{SUMMARY_PATH} not found — run "
                "`python -m src.inference.build_dashboard_summary` first."
            )
        else:
            render_explore_tab(load_summary())
    with tab_try_it:
        render_try_it_tab()


if __name__ == "__main__":
    main()
