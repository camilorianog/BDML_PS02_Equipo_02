# ============================================================
# 00_rundirectory.R
# Modelos de Clasificación — configuración + pipeline
# ============================================================
#
# MECA 4107 — Big Data y Machine Learning para Economía Aplicada
# Universidad de los Andes | 2026-10
#
# Equipo 02:
#   · Jose A. Rincón S.    — 202013328
#   · Juan C. Riaño        — 202013305
#   · Lucas Rodriguez      — 202021985
#   · Santiago González    — 202110234
#
# Descripción:
#   Script maestro del pipeline de predicción de pobreza.
#   Corre en orden: limpieza → features → EDA → modelos.
#   Fuente: DANE MESE 2018, Bogotá.
#   Métrica objetivo: F1-score (CV 5 folds).
# ============================================================

cat("\n")
cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║   MECA 4107 · Problem Set 02 · Equipo 02                 ║\n")
cat("║   Predicción de Pobreza — DANE MESE 2018, Bogotá         ║\n")
cat("║                                                          ║\n")
cat("║   Jose A. Rincón S.  ·  Juan C. Riaño                    ║\n")
cat("║   Lucas Rodriguez    ·  Santiago Gonzalez.               ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n")
cat(sprintf("  Inicio: %s\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

# ============================================================
# PAQUETES
# ============================================================

if (!require("pacman", quietly = TRUE)) install.packages("pacman")

pacman::p_load(
  # Entorno
  here, tictoc, jsonlite, httr, reticulate,
  
  # Manipulación de datos
  tidyverse, janitor, skimr, patchwork, gt,
  
  # Modelado
  caret, glmnet, naivebayes, ranger, xgboost, lightgbm, bonsai,
  
  # Métricas
  yardstick, MLmetrics,
  
  # Visualización
  ggplot2
)

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================

SEED     <- 202601
CV_FOLDS <- 5

set.seed(SEED)

# Grid Elastic Net
EN_GRID <- expand.grid(
  alpha  = seq(0.1, 0.9, by = 0.01),
  lambda = 10^seq(-4, 1, length = 20)
)

# ============================================================
# RUTAS
# ============================================================

paths <- list(
  root        = here::here(),
  raw         = here("00_data", "00_raw"),
  processed   = here("00_data", "01_processed"),
  prep        = here("01_R",    "00_prep"),
  feat        = here("01_R",    "01_feat"),
  functions   = here("01_R",    "02_functions"),
  models      = here("02_models"),
  classes     = here("02_models", "00_classes"),
  submissions = here("02_models", "01_submissions"),
  figures     = here("04_outputs", "figures"),
  tables      = here("04_outputs", "tables")
)

invisible(lapply(paths, dir.create, recursive = TRUE, showWarnings = FALSE))

invisible(lapply(
  c("01_Base_models", "02_LPM", "03_Logit", "04_Elastic_Net",
    "05_CART", "06_Random_Forest", "07_Boosting", "08_Naive_Bayes"),
  function(d) dir.create(file.path(paths$submissions, d),
                         recursive = TRUE, showWarnings = FALSE)
))

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

source(here(paths$functions, "00_optimizar_threshold.R"))
source(here(paths$functions, "01_guardar_modelo.R"))
source(here(paths$functions, "02_generar_submission.R"))

# ============================================================
# PIPELINE
# ============================================================

tic("Pipeline completo")

# --- [1] Limpieza y preparación -----------------------------
cat("─────────────────────────────────────────────────────────\n")
cat("  [1/5] Limpieza y preparación de datos\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Limpieza")
source(here(paths$prep, "00_clean.R"))
toc(log = TRUE)

# --- [2] Feature engineering --------------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [2/5] Feature engineering\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Features")
source(here(paths$feat, "00_features.R"))
toc(log = TRUE)

# --- [3] EDA ------------------------------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [3/5] Análisis exploratorio (EDA)\n")
cat("─────────────────────────────────────────────────────────\n")
tic("EDA")
source(here(paths$prep, "01_eda.R"))
toc(log = TRUE)

# --- [4] Modelos de probabilidad ----------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [4/5] Modelos de probabilidad\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Modelos probabilidad")
source(here(paths$classes, "01_Base_models.R"))
source(here(paths$classes, "02_LPM.R"))
source(here(paths$classes, "03_Logit.R"))
source(here(paths$classes, "04_Elastic_Net.R"))
toc(log = TRUE)

# --- [5] Modelos basados en árboles -------------------------
cat("\n─────────────────────────────────────────────────────────\n")
cat("  [5/5] Modelos basados en árboles\n")
cat("─────────────────────────────────────────────────────────\n")
tic("Modelos árboles")
source(here(paths$classes, "05_CART.R"))
source(here(paths$classes, "06_Random_Forest.R"))
source(here(paths$classes, "07_Boosting.R"))
source(here(paths$classes, "08_Naive_Bayes.R"))
toc(log = TRUE)

# ============================================================
# RESUMEN FINAL
# ============================================================

cat("╔══════════════════════════════════════════════════════════╗\n")
cat("║   Pipeline completado                                    ║\n")
cat("╚══════════════════════════════════════════════════════════╝\n")
cat("  Tiempos por etapa:\n")
tic.log(format = TRUE) |> unlist() |> cat(sep = "\n")
cat("\n")
toc()
cat(sprintf("\n  Fin: %s\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
