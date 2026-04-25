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
| Santiago Gonzalez| 202156304 |

---

## Descripción

Este repositorio contiene el análisis del **Problem Set 02** del curso de Big Data y Machine Learning para Economía Aplicada. El objetivo es predecir si un hogar es pobre utilizando datos de la Gran Encuesta Integrada de Hogares (GEIH / MESE) 2018 para Bogotá, evaluando modelos con la métrica **F1-score** en una competencia de Kaggle.

El problema está motivado por un reto del Banco Mundial: medir pobreza mediante encuestas completas es costoso, y un modelo predictivo robusto permite diseñar instrumentos de focalización más cortos y eficientes.

---

## Instrucciones de Replicación

> **Requisito:** R ≥ 4.3.0. Los datos crudos deben descargarse desde Kaggle antes de correr el pipeline.

```r
# 1. Descargar datos (requiere kaggle CLI configurado)
# kaggle competitions download -c uniandes-bdml-2026-10-ps2
# Descomprimir en 00_data/00_raw/

# 2. Correr el pipeline completo
source("00_rundirectory.R")
```

El script maestro ejecuta secuencialmente:

1. **Limpieza y preparación** de datos crudos (`00_clean.R`)
2. **Feature engineering** — construcción de variables desde nivel individual (`00_features.R`)
3. **Modelos de probabilidad** — Base, LPM, Logit, Elastic Net
4. **Reducción de variables** para los modelos de árbol (`00_reduction.R`)
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
│   │   └── 00_clean.R         # Limpieza, merge hogares–personas y reporte de missings
│   ├── 01_feat/
│   │   └── 00_features.R      # Feature engineering desde nivel individual
│   ├── 02_functions/
│   │   ├── 00_optimizar_threshold.R   # Optimización de threshold para F1
│   │   ├── 01_guardar_modelo.R        # Serialización de modelos (.rds)
│   │   └── 02_generar_submission.R    # Generación de CSV para Kaggle
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

[Kaggle — uniandes-bdml-2026-10-ps2](https://www.kaggle.com/competitions/uniandes-bdml-2026-10-ps2)

Los datos provienen de la **Gran Encuesta Integrada de Hogares (GEIH / MESE)** del DANE para Bogotá, 2018. Se componen de cuatro archivos CSV que se unen mediante la variable `id`:

| Archivo | Nivel | Obs. (aprox.) |
|---------|-------|---------------|
| `hogares_train.csv` | Hogar | Set de entrenamiento |
| `hogares_test.csv` | Hogar | Set de prueba (sin `pobre`) |
| `personas_train.csv` | Individuo | Set de entrenamiento |
| `personas_test.csv` | Individuo | Set de prueba (variables restringidas) |

> **Nota:** Varias variables del archivo de personas **no están disponibles en el set de prueba** — restricción intencional del PS que fuerza un feature engineering real desde variables observables.

### Variable Objetivo

`pobre = 1` si el ingreso per cápita del hogar (`Li`) es inferior a la línea de pobreza (`Lp`); `0` en caso contrario. La clase positiva (hogares pobres) es minoría, lo que genera **desbalance de clases** y es el principal desafío metodológico del problema.

### Variables Construidas

El script `00_features.R` colapsa la información de personas al nivel de hogar, construyendo las siguientes variables agrupadas en cuatro categorías con justificación económica:

**Demografía del hogar** — carga de dependientes sobre los que generan ingreso:
- `n_miembros`: número total de miembros
- `n_menores`: número de menores de 18 años
- `n_adultos_mayores`: número de mayores de 60 años
- `prop_dependientes`: fracción de miembros económicamente dependientes
- `ratio_ninos_adultos`: razón niños / adultos en edad de trabajar

**Capital humano** — capacidad de generación de ingresos (Mincer, 1974):
- `educ_max_hogar`: años de educación del miembro más educado
- `educ_media_hogar`: promedio de años de educación del hogar
- `educ_min_hogar`: años de educación del miembro menos educado

**Jefe de hogar** — proxy de ingreso permanente del hogar:
- `jefe_edad`: edad del jefe de hogar
- `jefe_sexo_mujer`: indicador de jefatura femenina
- `jefe_educ`: años de educación del jefe
- `jefe_trabaja`: indicador de si el jefe está ocupado

**Situación laboral** — principal fuente de ingreso corriente:
- `n_ocupados`: número de miembros ocupados
- `prop_ocupados`: fracción del hogar que trabaja
- `n_desocupados`: número de miembros buscando empleo
- `ratio_empleo_pea`: razón ocupados / población en edad de trabajar (18–60)

---

## Metodología

Se comparan 8 especificaciones de modelos de clasificación evaluadas con **F1-score** mediante validación cruzada de **5 folds**:

| # | Modelo | Librería | Notas |
|---|--------|----------|-------|
| 1 | Base (umbral fijo) | — | Benchmark de referencia |
| 2 | LPM | `lm` | Probabilidad lineal |
| 3 | Logit | `glm` | Regresión logística |
| 4 | Elastic Net | `glmnet` | Regularización L1 + L2 |
| 5 | CART | `caret` / `rpart` | Árbol de decisión |
| 6 | Random Forest | `ranger` | Ensamble de árboles |
| 7 | Boosting | `xgboost` / `lightgbm` | Gradiente boosting — **mejor modelo** |
| 8 | Naive Bayes | `naivebayes` | Clasificador probabilístico |

### Optimización de Threshold

El threshold de clasificación **no se fija en 0.5**. Se optimiza post-entrenamiento mediante búsqueda en grilla sobre `[0.1, 0.9]` para maximizar el F1 en el set de validación, compensando el desbalance de clases.

### Mejor Modelo

El modelo con mejor desempeño en Kaggle es **XGBoost** con los siguientes hiperparámetros:

```
nrounds          = 500
max_depth        = 6
eta              = 0.05
subsample        = 0.8
colsample_bytree = 0.8
min_child_weight = 5
objective        = binary:logistic
```

> El F1-score público reportado en el leaderboard de Kaggle es ~0.71–0.73. El leaderboard público usa solo el 20% del test; el score privado (80%) determina el ranking final.

---

## Software

```
R version 4.3.0 o superior
```

### Paquetes Requeridos

Instalados y cargados automáticamente mediante `pacman::p_load()`:

| Categoría | Paquetes |
|-----------|----------|
| Entorno | `here`, `tictoc`, `jsonlite`, `httr`, `reticulate` |
| Datos | `tidyverse`, `janitor`, `skimr`, `data.table` |
| Modelado | `caret`, `glmnet`, `naivebayes`, `ranger`, `xgboost`, `lightgbm`, `bonsai`, `rpart`, `rpart.plot` |
| Desbalance | `themis` |
| Métricas | `yardstick`, `MLmetrics`, `pROC` |
| Visualización | `ggplot2`, `ggthemes`, `patchwork`, `viridis` |
| Tablas | `gt`, `kableExtra`, `stargazer` |

> Si `pacman` no está instalado: `install.packages("pacman")`

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
