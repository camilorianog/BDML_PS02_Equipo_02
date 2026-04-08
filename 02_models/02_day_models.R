# ============================================================
# 02_day_models.R
# Día 02 — Elastic net
# ============================================================

DIA <- "02"
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
# MODELO 1 — Elastic net full grid
# ============================================================
cat("\n>>> [1/2] Elastic net full grid...\n")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = EN_GRID
)

opt1 <- optimizar_threshold(m1, train, train$pobre)

nombre_m1 <- paste0(
  "EN_lambda_", m1$bestTune$lambda,
  "_alpha_",    m1$bestTune$alpha
)

guardar_modelo(m1, nombre_m1, DIA, dir_dia, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, DIA, nombre_m1)

# ============================================================
# MODELO 2 — Elastic net full grid con Pre-Process
# ============================================================
cat("\n>>> [2/2] Elastic net full grid con pre process...\n")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  preProcess = c("center", "scale"),
  tuneGrid  = EN_GRID
)

opt2 <- optimizar_threshold(m2, train, train$pobre)

nombre_m2 <- paste0(
  "EN_lambda_", m2$bestTune$lambda,
  "_alpha_",    m2$bestTune$alpha
)

guardar_modelo(m2, nombre_m2, DIA, dir_dia, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, DIA, nombre_m2)

# Generó exactamente mismo modelo al pasado

# ============================================================
# MODELO 3 — Logística baseline con limpieza adicional de datos
# ============================================================
cat("\n>>> [1] Logística baseline...\n")
set.seed(SEED)

m3 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "AUC"
)

opt3 <- optimizar_threshold(m3, train, train$pobre)
nombre_m3 <- "logit_baseline_v2"

guardar_modelo(m3, nombre_m3, DIA, dir_dia, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, DIA, nombre_m3)

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
