# Problem Set 02: Clasificación de Pobreza

## Grupo 2 | Big Data and Machine Learning para Economía Aplicada

**MECA 4107** — Universidad de los Andes — 2026-10

---

## Autores

| Nombre           | Código    |
|------------------|-----------|
| Jose A. Rincon S | 202013328 |
| Juan C. Riaño    | 202013305 |
| Lucas Rodriguez  | 202021985 |

---

## Descripción

Este repositorio contiene el análisis del **Problem Set 02** del curso de Big Data y Machine Learning para Economía Aplicada. El objetivo es predecir si un hogar es pobre utilizando datos de la Gran Encuesta Integrada de Hogares (GEIH) 2018 para Bogotá, evaluando modelos con la métrica **F1-score**.

---

## Instrucciones de Replicación

> **Requisito:** R ≥ 4.3.0. Los datos crudos deben descargarse desde Kaggle antes de correr el pipeline.

```r
# 1. Descargar datos (requiere kaggle CLI configurado)
# kaggle competitions download -c <slug-de-la-competencia>
# Descomprimir en 00_data/00_raw/

# 2. Correr el pipeline completo
source("00_rundirectory.R")
```

El script maestro ejecuta secuencialmente:

1. **Limpieza y preparación** de datos crudos (`00_clean.R`)
2. **Feature engineering** — construcción de variables (`00_features.R`)
3. **Modelos de probabilidad** — Base, LPM, Logit, Elastic Net
4. **Reducción de variables** para los modelos de árbol
5. **Modelos basados en árboles** — CART, Random Forest, Boosting, Naive Bayes

---

## Estructura del Repositorio

```
.
├── 00_rundirectory.R          # Script maestro — punto de entrada
├── BDML_PS02_Equipo_02.Rproj  # Proyecto de RStudio
│
├── 00_data/
│   ├── 00_raw/                # Datos crudos (ignorados en git)
│   └── 01_processed/          # Datos procesados (ignorados en git)
│
├── 01_R/
│   ├── 00_prep/
│   │   └── 00_clean.R         # Limpieza y preparación
│   ├── 01_feat/
│   │   └── 00_features.R      # Feature engineering
│   ├── 02_functions/
│   │   ├── 00_optimizar_threshold.R
│   │   ├── 01_guardar_modelo.R
│   │   └── 02_generar_submission.R
│   └── 03_reduced/
│       └── 00_reduction.R     # Base reducida para modelos de árbol
│
└── 02_models/
    ├── 00_classes/
    │   ├── 01_Base_models.R
    │   ├── 02_LPM.R
    │   ├── 03_Logit.R
    │   ├── 04_Elastic_Net.R
    │   ├── 05_CART.R
    │   ├── 06_Random_Forest.R
    │   ├── 07_Boosting.R
    │   └── 08_Naive_Bayes.R
    └── 01_submissions/        # Archivos de submission (ignorados en git)
```

---

## Datos

### Fuente

[Kaggle — MECA 4107 PS02]() *(actualizar con el slug real)*

### Variables Principales

<!-- Completar con las variables del dataset de Kaggle -->

### Construcción de la Muestra

<!-- Describir filtros aplicados: hogares en Bogotá, GEIH 2018, etc. -->

### Variables Construidas

<!-- Listar variables creadas en el feature engineering -->

---

## Metodología

Se comparan 8 especificaciones de modelos de clasificación evaluadas con **F1-score** mediante validación cruzada de **5 folds**:

| # | Modelo         | Librería         |
|---|----------------|------------------|
| 1 | Base (umbral fijo) | —            |
| 2 | LPM            | `lm`             |
| 3 | Logit          | `glm`            |
| 4 | Elastic Net    | `glmnet`         |
| 5 | CART           | `caret` / `rpart`|
| 6 | Random Forest  | `ranger`         |
| 7 | Boosting       | `xgboost` / `lightgbm` |
| 8 | Naive Bayes    | `naivebayes`     |

El threshold de clasificación se optimiza post-entrenamiento para maximizar el F1 en el set de validación.

---

## Software

```
R version 4.3.0 o superior
```

### Paquetes Requeridos

Instalados y cargados automáticamente mediante `pacman::p_load()`:

| Categoría     | Paquetes                                          |
|---------------|---------------------------------------------------|
| Entorno       | `here`, `tictoc`, `jsonlite`, `httr`, `reticulate`|
| Datos         | `tidyverse`, `janitor`, `skimr`                   |
| Modelado      | `caret`, `glmnet`, `naivebayes`, `ranger`, `xgboost`, `lightgbm`, `bonsai` |
| Métricas      | `yardstick`, `MLmetrics`                          |
| Visualización | `ggplot2`                                         |

---

## Referencias

- Sarmiento-Barbieri, I. (2026). *Big Data and Machine Learning for Applied Economics*. Universidad de los Andes.
- DANE. *Gran Encuesta Integrada de Hogares (GEIH) 2018*.
- Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
- Blau, F. & Kahn, L. (2017). The Gender Wage Gap. *Journal of Economic Literature*.
- Kleven, H. et al. (2019). Child Penalties Across Countries. *AER Insights*.

---

## Contacto

Para preguntas sobre este repositorio, contactar a los autores mencionados al inicio.
