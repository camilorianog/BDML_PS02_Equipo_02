# ============================================================
# Boosting.R
# Gradient Boosting models (XGBoost)
# ============================================================

TIPO       <- "Boosting"
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
# MODELO 1 — XGBoost default
# ============================================================
cat("\n>>> [boosting - 1/3] XGBoost default...\n")
tic("XGBoost default")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "xgbTree",
  trControl = ctrl,
  metric    = "AUC"
)

opt1      <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- paste0("XGB_depth_", m1$bestTune$max_depth,
                    "_eta_",      m1$bestTune$eta)
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO, nombre_m1)
toc()

# ============================================================
# MODELO 2 — XGBoost grid reducido
# ============================================================
cat("\n>>> [boosting - 2/3] XGBoost grid reducido...\n")
tic("XGBoost grid reducido")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "xgbTree",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    nrounds          = c(100, 300),
    max_depth        = c(3, 6),
    eta              = c(0.01, 0.1),
    gamma            = 0,
    colsample_bytree = 0.7,
    min_child_weight = 1,
    subsample        = 0.8
  )
)

opt2      <- optimizar_threshold(m2, train, train$pobre)
nombre_m2 <- paste0("XGB_depth_", m2$bestTune$max_depth,
                    "_eta_",      m2$bestTune$eta,
                    "_rounds_",   m2$bestTune$nrounds)
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# MODELO 3 — XGBoost grid amplio
# ============================================================
cat("\n>>> [boosting - 3/3] XGBoost grid amplio...\n")
tic("XGBoost grid amplio")
set.seed(SEED)

m3 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "xgbTree",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    nrounds          = c(100, 300, 500),
    max_depth        = c(3, 6, 9),
    eta              = c(0.01, 0.05, 0.1),
    gamma            = c(0, 1),
    colsample_bytree = c(0.5, 0.7),
    min_child_weight = c(1, 5),
    subsample        = c(0.7, 0.9)
  )
)

opt3      <- optimizar_threshold(m3, train, train$pobre)
nombre_m3 <- paste0("XGB_depth_", m3$bestTune$max_depth,
                    "_eta_",      m3$bestTune$eta,
                    "_rounds_",   m3$bestTune$nrounds)
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, TIPO, nombre_m3)
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen Boosting\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre)"))
rm(ctrl, dir_modelo, TIPO)
gc()