# ============================================================
# Random_Forest.R
# Random Forest models
# ============================================================

TIPO       <- "Random_Forest"
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

# --- Parámetro base -----------------------------------------
p            <- ncol(train) - 2  # descontar id y pobre
mtry_default <- floor(sqrt(p))

# ============================================================
# MODELO 1 — RF mtry default
# ============================================================
cat("\n>>> [rf - 1/2] Random Forest mtry default...\n")
tic("RF mtry default")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rf",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(mtry = mtry_default)
)

opt1      <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- paste0("RF_mtry_", m1$bestTune$mtry)
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO, nombre_m1)
toc()

# ============================================================
# MODELO 2 — RF grid reducido
# ============================================================
cat("\n>>> [rf - 2/2] Random Forest grid reducido...\n")
tic("RF grid reducido")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rf",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    mtry = c(floor(mtry_default / 2), mtry_default, floor(mtry_default * 2))
  )
)

opt2      <- optimizar_threshold(m2, train, train$pobre)
nombre_m2 <- paste0("RF_mtry_", m2$bestTune$mtry)
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen Random Forest\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre)"))
rm(ctrl, dir_modelo, TIPO, p, mtry_default)
gc()