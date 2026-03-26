# ============================================================
# 00_day_models.R
# Día 00 — Logística baseline + Elastic net
# ============================================================

DIA <- "00"
dir_dia <- here(paths$models, paste0( DIA, "_day"))
dir.create(dir_dia, recursive = TRUE, showWarnings = FALSE)

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
# MODELO 1 — Logística baseline
# ============================================================
cat("\n>>> [1/5] Logística baseline...\n")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "AUC"
)

opt1 <- optimizar_threshold(m1, train, train$pobre)
guardar_modelo(m1, "01_logit_baseline", DIA, dir_dia, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, DIA)

# ============================================================
# MODELO 2 — Elastic net ridge (alpha = 0)
# ============================================================
cat("\n>>> [2/5] Elastic net ridge...\n")
set.seed(SEED)

m2 <- train(
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

opt2 <- optimizar_threshold(m2, train, train$pobre)
guardar_modelo(m2, "02_elastic_ridge", DIA, dir_dia, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, "02_elastic_ridge")

# ============================================================
# MODELO 3 — Elastic net lasso (alpha = 1)
# ============================================================
cat("\n>>> [3/5] Elastic net lasso...\n")
set.seed(SEED)

m3 <- train(
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

opt3 <- optimizar_threshold(m3, train, train$pobre)
guardar_modelo(m3, "03_elastic_lasso", DIA, dir_dia, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, "03_elastic_lasso", DIA)

# ============================================================
# MODELO 4 — Elastic net mix (alpha = 0.5)
# ============================================================
cat("\n>>> [4/5] Elastic net mix...\n")
set.seed(SEED)

m4 <- train(
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

opt4 <- optimizar_threshold(m4, train, train$pobre)
guardar_modelo(m4, "04_elastic_mix", DIA, dir_dia, opt4$threshold, opt4$f1)
generar_submission(m4, test, opt4$threshold, "04_elastic_mix", DIA)

# ============================================================
# MODELO 5 — Elastic net full grid
# ============================================================
cat("\n>>> [5/5] Elastic net full grid...\n")
set.seed(SEED)

m5 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = EN_GRID
)

opt5 <- optimizar_threshold(m5, train, train$pobre)
guardar_modelo(m5, "05_elastic_full", DIA, dir_dia, opt5$threshold, opt5$f1)
generar_submission(m5, test, opt5$threshold, "05_elastic_full", DIA)

# ============================================================
# RESUMEN DEL DÍA
# ============================================================
cat("\n======================================================\n")
cat("  Resumen día", DIA, "\n")
cat("======================================================\n")

read.csv(here(paths$models, "log.csv")) |>
  filter(dia == DIA) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(m1, m2, m3, m4, m5, ctrl,
   opt1, opt2, opt3, opt4, opt5,
   dir_dia, DIA)
gc()