# CMS Fraud Detection

## Objetivo

Detectar posible fraude entre proveedores de salud a partir de sus relaciones
financieras con farmacéuticas y fabricantes de dispositivos médicos. El
dataset de pagos no viene etiquetado, así que la etiqueta de fraude se
construye cruzándolo con la lista oficial de proveedores excluidos por
fraude. El proyecto compara dos enfoques de modelado para evaluar cuál
captura mejor los patrones de fraude:

- **Modelado supervisado**, usando la etiqueta derivada del cruce.
- **Detección de anomalías no supervisada**, sin usar la etiqueta durante el
  entrenamiento (la etiqueta solo se usa después, para validar qué tan bien
  las anomalías detectadas coinciden con los casos de exclusión reales).

## Fuentes de datos

- **CMS Open Payments** (cms.gov): pagos de farmacéuticas y fabricantes de
  dispositivos médicos a médicos y hospitales en EE. UU. Es la fuente de
  features: montos, tipos de pago, entidades pagadoras, especialidades, etc.
- **LEIE** (List of Excluded Individuals/Entities): lista de proveedores
  excluidos de programas federales de salud por fraude u otras causas. Es la
  fuente de la etiqueta.

### Cruce

Los proveedores de Open Payments se cruzan contra la LEIE (típicamente por
nombre/NPI y fecha de exclusión) para marcar qué proveedores estaban
excluidos por fraude. Este cruce produce la columna objetivo
`TARGET_COLUMN` (`is_excluded`, definida en [src/config.py](src/config.py))
y se implementa en
[src/data/clean_data.py](src/data/clean_data.py) (`label_fraud`). Al ser un
cruce con una lista de exclusión, el dataset resultante es fuertemente
desbalanceado: hay que tenerlo en cuenta al elegir métricas (priorizar
precision/recall/PR-AUC sobre accuracy) y al comparar ambos enfoques.

## Convenciones del proyecto

- **Código modular en `src/`**: toda la lógica reutilizable (carga, limpieza,
  features, entrenamiento, evaluación, inferencia, visualización) vive en
  `src/`, organizada por responsabilidad (`data/`, `features/`, `models/`,
  `inference/`, `visualization/`). Nada de lógica de negocio duplicada en
  notebooks o en la app de Streamlit: ambos deben importar desde `src/`.
- **`notebooks/` solo para exploración**: se usan para EDA, prototipado y
  análisis puntuales, no para código de producción. Cualquier lógica que
  se vaya a reutilizar debe migrarse a `src/`.
- **Un test por módulo en `tests/`**: `tests/` espeja la estructura de
  `src/` (mismo árbol de carpetas, un archivo `test_*.py` por módulo). Al
  añadir o modificar un módulo en `src/`, actualizar su test correspondiente.
- **`models/`** guarda los artefactos entrenados (vía
  [src/models/model_io.py](src/models/model_io.py)); la promoción del mejor
  candidato a producción pasa por
  [src/models/promote_model.py](src/models/promote_model.py).
- Configuración y rutas centralizadas en
  [src/config.py](src/config.py) — no hardcodear rutas en otros módulos.
