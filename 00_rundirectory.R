# ============================================================
# 00_run.R
# Punto de entrada del proyecto — configuración + pipeline
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
  
  # Modelado
  caret,
  glmnet,
  
  # Utilidades
  yardstick,
  MLmetrics,
  
  #Gráficos
  ggplot2
)

# --- Gestión de funciones -----------------------------------

source(here("01_R", "02_functions", "00_optimizar_threshold.R"))
source(here("01_R", "02_functions", "01_guardar_modelo.R"))
source(here("01_R", "02_functions", "02_generar_submission.R"))

# --- Semilla ------------------------------------------------

SEED <- 202601

# --- CV -----------------------------------------------------

CV_FOLDS  <- 10
CV_METRIC <- "F1"

# --- Parámetros elastic net ---------------------------------

EN_GRID <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
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
  models      = here("02_models"),
  retired     = here("02_models", "99_retired"),
  submissions = here("03_submissions")
)

# --- Crear carpetas si no existen ---------------------------
invisible(lapply(paths, dir.create, recursive = TRUE, showWarnings = FALSE))

# Crear carpetas de modelos por tipo
model_dirs <- c("Base_models", "LPM", "Logit", "Elastic_Net",
                "CART", "Random_Forest", "Boosting", "Naive_Bayes")
invisible(lapply(model_dirs, function(d) {
  dir.create(here("02_models", d), recursive = TRUE, showWarnings = FALSE)
}))


# ============================================================
# PIPELINE
# ============================================================

tic("Pipeline completo")
cat("\n======================================================\n")
cat("  Iniciando pipeline\n")
cat("======================================================\n\n")

# --- 1. Preparación de datos --------------------------------

cat(">>> [1/4] Limpieza y preparación de datos...\n")
tic("Prep")
source(here("01_R", "00_prep", "00_clean.R"))
toc()

# --- 2. Feature engineering ---------------------------------

cat("\n>>> [2/4] Feature engineering...\n")
tic("Features")
source(here("01_R", "01_feat", "00_features.R"))
toc()

# --- 3. Modelado --------------------------------------------

cat("\n>>> [3/4] Entrenamiento de modelos...\n")
tic("Modelos")
source(here("02_models", "Base_models.R"))
source(here("02_models", "LPM.R"))
source(here("02_models", "Elastic_Net.R"))
source(here("02_models", "Random_Forest.R"))
source(here("02_models", "Logit.R"))
source(here("02_models", "CART.R"))
source(here("02_models", "Boosting.R"))
source(here("02_models", "Naive_Bayes.R"))
toc()

# --- 4. Submissions -----------------------------------------

cat("\n>>> [4/4] Generando submissions...\n")
tic("Submissions")

toc()



