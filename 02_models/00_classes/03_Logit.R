# ============================================================
# 03_Logit.R
# Logistic Regression models
# ============================================================

TIPO       <- "03_Logit"
dir_modelo <- here(paths$models, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

# --- Cargar datos -------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

train <- train |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# --- Control de entrenamiento -------------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = CV_FOLDS,
  classProbs      = TRUE,
  summaryFunction = prSummary,
  savePredictions = "final"
)

# ============================================================
# MODELO 1 — Logit baseline
# ============================================================
cat("\n>>> [logit - 1/2] Logit baseline...\n")
tic("Logit baseline")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "AUC"
)

opt1      <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- "logit_baseline"
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO, nombre_m1)
toc()

# ============================================================
# MODELO 2 — Logit con preprocesamiento
# ============================================================
cat("\n>>> [logit - 2/2] Logit preprocesado...\n")
tic("Logit preprocesado")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data       = train |> select(-id),
  method     = "glm",
  family     = binomial(link = "logit"),
  trControl  = ctrl,
  metric     = "AUC",
  preProcess = c("center", "scale")
)

opt2      <- optimizar_threshold(m2, train, train$pobre)
nombre_m2 <- "logit_scaled"
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen Logit\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre)"))
rm(ctrl, dir_modelo, TIPO)
gc()