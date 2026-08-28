"""App de Streamlit para explorar predicciones de fraude sobre CMS Open Payments."""

import streamlit as st

st.set_page_config(page_title="Detección de Fraude - CMS Open Payments", layout="wide")


def main() -> None:
    st.title("Detección de Fraude en CMS Open Payments")
    st.write("Compara el enfoque supervisado (etiquetas LEIE) vs. detección de anomalías no supervisada.")


if __name__ == "__main__":
    main()
