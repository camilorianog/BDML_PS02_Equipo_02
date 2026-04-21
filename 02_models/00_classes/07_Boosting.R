# ============================================================
# 07_Boosting.R
# Gradient Boosting models (XGBoost + LightGBM)
# ============================================================

TIPO       <- "07_Boosting"
dir_modelo <- here(paths$submissions, TIPO)
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
cat("\n>>> [boosting - 1/6] XGBoost default...\n")
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
cat("\n>>> [boosting - 2/6] XGBoost grid reducido...\n")
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
nombre_m2 <- paste0("XGB_depth_",  m2$bestTune$max_depth,
                    "_eta_",       m2$bestTune$eta,
                    "_rounds_",    m2$bestTune$nrounds)
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# MODELO 3 — LightGBM default
# ============================================================
cat("\n>>> [boosting - 3/5] LightGBM default...\n")
tic("LightGBM default")
set.seed(SEED)

m3 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "lgb",
  trControl = ctrl,
  metric    = "AUC"
)

opt3      <- optimizar_threshold(m3, train, train$pobre)
nombre_m3 <- paste0("LGB_depth_",  m3$bestTune$max_depth,
                    "_leaves_",    m3$bestTune$num_leaves)
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt4$threshold, TIPO, nombre_m3)
toc()

# ============================================================
# MODELO 4 — LightGBM grid reducido
# ============================================================
cat("\n>>> [boosting - 4/5] LightGBM grid reducido...\n")
tic("LightGBM grid reducido")
set.seed(SEED)

m4 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "lgb",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    num_leaves       = c(31, 63),
    max_depth        = c(-1, 6),
    learning_rate    = c(0.05, 0.1),
    n_iter           = c(100, 300),
    min_data_in_leaf = 20,
    feature_fraction = 0.8
  )
)

opt4      <- optimizar_threshold(m4, train, train$pobre)
nombre_m4 <- paste0("LGB_depth_",  m4$bestTune$max_depth,
                    "_leaves_",    m4$bestTune$num_leaves,
                    "_lr_",        m4$bestTune$learning_rate,
                    "_iter_",      m4$bestTune$n_iter)
guardar_modelo(m4, nombre_m4, TIPO, dir_modelo, opt4$threshold, opt4$f1)
generar_submission(m4, test, opt4$threshold, TIPO, nombre_m4)
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