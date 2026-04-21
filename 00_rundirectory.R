# ============================================================
# 00_rundirectory.R
# Modelos de Clasificación — configuración + pipeline
# Equipo 02 | MECA 4107 | Universidad de los Andes | 2026-10
# ============================================================

# --- Gestión de paquetes ------------------------------------

if (!require("pacman", quietly = TRUE)) install.packages("pacman")

pacman::p_load(
  # Entorno
  here,
  tictoc,
  jsonlite,
  httr,
  reticulate,
  
  # Manipulación de datos
  tidyverse,
  janitor,
  skimr,
  dplyr,
  
  # Modelado
  caret,
  glmnet,
  naivebayes,
  ranger,
  xgboost,
  lightgbm,
  bonsai,
  themis,
  
  # Métricas
  yardstick,
  MLmetrics,
  
  # Visualización
  ggplot2
)

# --- Semilla ------------------------------------------------

SEED <- 202601
set.seed(SEED)

# --- Parámetros globales de CV ------------------------------

CV_FOLDS  <- 5
CV_METRIC <- "F1"

# --- Grid elastic net ---------------------------------------

EN_GRID <- expand.grid(
  alpha  = seq(0.1, 0.9, by = 0.01),
  lambda = 10^seq(-4, 1, length = 20)
)

# --- Rutas --------------------------------------------------

paths <- list(
  root        = here::here(),
  raw         = here("00_data", "00_raw"),
  processed   = here("00_data", "01_processed"),
  prep        = here("01_R",    "00_prep"),
  feat        = here("01_R",    "01_feat"),
  functions   = here("01_R",    "02_functions"),
  reduced     = here("01_R",    "03_reduced"),
  models      = here("02_models"),
  classes     = here("02_models", "00_classes"),
  submissions = here("02_models", "01_submissions")
)

# --- Crear estructura de carpetas ---------------------------

invisible(lapply(paths, dir.create, recursive = TRUE, showWarnings = FALSE))

model_dirs <- c(
  "01_Base_models",
  "02_LPM",
  "03_Logit",
  "04_Elastic_Net",
  "05_CART",
  "06_Random_Forest",
  "07_Boosting",
  "08_Naive_Bayes"
)

invisible(lapply(model_dirs, function(d) {
  dir.create(file.path(paths$submissions, d), recursive = TRUE, showWarnings = FALSE)
}))

# --- Cargar funciones auxiliares ----------------------------

source(here("01_R", "02_functions", "00_optimizar_threshold.R"))
source(here("01_R", "02_functions", "01_guardar_modelo.R"))
source(here("01_R", "02_functions", "02_generar_submission.R"))

# ============================================================
# PIPELINE
# ============================================================


tic("Pipeline completo")

cat("\n")
cat("============================================================\n")
cat("  MECA 4107 | Problem Set 02 | Equipo 02\n")
cat("  Iniciando pipeline —", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")

# --- 1. Preparación de datos --------------------------------

cat(">>> Limpieza y preparación de datos...\n")
tic("  Prep")
source(here("01_R", "00_prep", "00_clean.R"))
toc(log = TRUE)

# --- 2. Feature engineering ---------------------------------

cat("\n>>> Feature engineering...\n")
tic("  Features")
source(here("01_R", "01_feat", "00_features.R"))
toc(log = TRUE)

# --- 3. Modelos de probabilidad -----------------------------

cat("\n>>> Entrenamiento de modelos...\n")

cat("  · Modelos de probabilidad\n")
tic("  Modelos probabilidad")
source(here("02_models", "00_classes", "01_Base_models.R"))
source(here("02_models", "00_classes", "02_LPM.R"))
source(here("02_models", "00_classes", "03_Logit.R"))
source(here("02_models", "00_classes", "04_Elastic_Net.R"))
toc(log = TRUE)

# --- 3.a Base reducida para contraste -----------------------

cat("  · Reducción de variables\n")
source(here("01_R", "03_reduced", "00_reduction.R"))

# --- 3.b Modelos basados en árboles -------------------------

cat("  · Modelos basados en árboles\n")
tic("  Modelos árboles")
source(here("02_models", "00_classes", "05_CART.R"))
source(here("02_models", "00_classes", "06_Random_Forest.R"))
source(here("02_models", "00_classes", "07_Boosting.R"))
source(here("02_models", "00_classes", "08_Naive_Bayes.R"))
toc(log = TRUE)

# --- 4. Resumen final ---------------------------------------

cat("\n>>> Pipeline finalizado.\n\n")
cat("------------------------------------------------------------\n")
cat("Tiempos por etapa:\n")
tic.log(format = TRUE) |> unlist() |> cat(sep = "\n")
cat("------------------------------------------------------------\n")
toc() # Pipeline completo
cat("\n")
