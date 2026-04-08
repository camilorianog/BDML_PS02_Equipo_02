# ============================================================
# elastic_net.R
# Modelos Elastic Net (Ridge, Lasso, Mix, Full Grid)
# ============================================================

TIPO       <- "Elastic_Net"
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
# MODELO 1 — Ridge (alpha = 0)
# ============================================================
cat("\n>>> [elastic_net - 1/4] Ridge...\n")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    alpha  = 0,
    lambda = 10^seq(-4, 1, length = 30)
  )
)

opt1   <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- paste0("EN_ridge_lambda_", round(m1$bestTune$lambda, 6))
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO)

# ============================================================
# MODELO 2 — Lasso (alpha = 1)
# ============================================================
cat("\n>>> [elastic_net - 2/4] Lasso...\n")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    alpha  = 1,
    lambda = 10^seq(-4, 1, length = 30)
  )
)

opt2      <- optimizar_threshold(m2, train, train$pobre)
nombre_m2 <- paste0("EN_lasso_lambda_", round(m2$bestTune$lambda, 6))
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO)

# ============================================================
# MODELO 3 — Mix (alpha = 0.5)
# ============================================================
cat("\n>>> [elastic_net - 3/4] Mix...\n")
set.seed(SEED)

m3 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    alpha  = 0.5,
    lambda = 10^seq(-4, 1, length = 30)
  )
)

opt3      <- optimizar_threshold(m3, train, train$pobre)
nombre_m3 <- paste0("EN_mix_lambda_", round(m3$bestTune$lambda, 6))
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, TIPO)

# ============================================================
# MODELO 4 — Full grid
# ============================================================
cat("\n>>> [elastic_net - 4/4] Full grid...\n")
set.seed(SEED)

m4 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = EN_GRID
)

opt4      <- optimizar_threshold(m4, train, train$pobre)
nombre_m4 <- paste0("EN_full_lambda_", round(m4$bestTune$lambda, 6),
                    "_alpha_",         m4$bestTune$alpha)
guardar_modelo(m4, nombre_m4, TIPO, dir_modelo, opt4$threshold, opt4$f1)
generar_submission(m4, test, opt4$threshold, TIPO)

# ============================================================
# MODELO 5 — Full grid (Pre-processed)
# ============================================================
cat("\n>>> [elastic_net - 5/5] Full grid Pre-Process...\n")
set.seed(SEED)

m5 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  preProcess = c("center", "scale"),
  tuneGrid  = EN_GRID
)

opt5      <- optimizar_threshold(m5, train, train$pobre)
nombre_m5 <- paste0("EN_full_lambda_", round(m5$bestTune$lambda, 6),
                    "_alpha_",         m5$bestTune$alpha)
guardar_modelo(m5, nombre_m4, TIPO, dir_modelo, opt5$threshold, opt5$f1)
generar_submission(m5, test, opt4$threshold, TIPO)

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen elastic_net\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(m1, m2, m3, m4,m5, ctrl,
   opt1, opt2, opt3, opt4,
   nombre_m1, nombre_m2, nombre_m3, nombre_m4,
   dir_modelo, TIPO)
gc()
