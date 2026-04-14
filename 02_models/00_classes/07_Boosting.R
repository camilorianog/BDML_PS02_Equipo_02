# ============================================================
# 07_Boosting.R
# Gradient Boosting models (XGBoost + LightGBM)
# ============================================================

TIPO       <- "07_Boosting"
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
# MODELO 3 — XGBoost grid amplio
# ============================================================
cat("\n>>> [boosting - 3/6] XGBoost grid amplio...\n")
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
nombre_m3 <- paste0("XGB_depth_",  m3$bestTune$max_depth,
                    "_eta_",       m3$bestTune$eta,
                    "_rounds_",    m3$bestTune$nrounds)
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, TIPO, nombre_m3)
toc()

# ============================================================
# MODELO 4 — LightGBM default
# ============================================================
cat("\n>>> [boosting - 4/6] LightGBM default...\n")
tic("LightGBM default")
set.seed(SEED)

m4 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "lgb",
  trControl = ctrl,
  metric    = "AUC"
)

opt4      <- optimizar_threshold(m4, train, train$pobre)
nombre_m4 <- paste0("LGB_depth_",  m4$bestTune$max_depth,
                    "_leaves_",    m4$bestTune$num_leaves)
guardar_modelo(m4, nombre_m4, TIPO, dir_modelo, opt4$threshold, opt4$f1)
generar_submission(m4, test, opt4$threshold, TIPO, nombre_m4)
toc()

# ============================================================
# MODELO 5 — LightGBM grid reducido
# ============================================================
cat("\n>>> [boosting - 5/6] LightGBM grid reducido...\n")
tic("LightGBM grid reducido")
set.seed(SEED)

m5 <- train(
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

opt5      <- optimizar_threshold(m5, train, train$pobre)
nombre_m5 <- paste0("LGB_depth_",  m5$bestTune$max_depth,
                    "_leaves_",    m5$bestTune$num_leaves,
                    "_lr_",        m5$bestTune$learning_rate,
                    "_iter_",      m5$bestTune$n_iter)
guardar_modelo(m5, nombre_m5, TIPO, dir_modelo, opt5$threshold, opt5$f1)
generar_submission(m5, test, opt5$threshold, TIPO, nombre_m5)
toc()

# ============================================================
# MODELO 6 — LightGBM grid amplio
# ============================================================
cat("\n>>> [boosting - 6/6] LightGBM grid amplio...\n")
tic("LightGBM grid amplio")
set.seed(SEED)

m6 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "lgb",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    num_leaves       = c(31, 63, 127),
    max_depth        = c(-1, 6, 9),
    learning_rate    = c(0.01, 0.05, 0.1),
    n_iter           = c(100, 300, 500),
    min_data_in_leaf = c(10, 20),
    feature_fraction = c(0.7, 0.9)
  )
)

opt6      <- optimizar_threshold(m6, train, train$pobre)
nombre_m6 <- paste0("LGB_depth_",  m6$bestTune$max_depth,
                    "_leaves_",    m6$bestTune$num_leaves,
                    "_lr_",        m6$bestTune$learning_rate,
                    "_iter_",      m6$bestTune$n_iter)
guardar_modelo(m6, nombre_m6, TIPO, dir_modelo, opt6$threshold, opt6$f1)
generar_submission(m6, test, opt6$threshold, TIPO, nombre_m6)
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