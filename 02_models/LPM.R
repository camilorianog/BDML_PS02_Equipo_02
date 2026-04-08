# ============================================================
# Modelos de Probabilidad Lineal (LPM)
# ============================================================

TIPO       <- "LPM"
dir_modelo <- here(paths$models, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

# --- Cargar datos -------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

# LPM usa variable numérica (0/1), no factor
# optimizar_threshold convierte internamente para confusionMatrix
train <- train |>
  mutate(pobre_num = as.numeric(pobre))

# --- Control de entrenamiento -------------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = CV_FOLDS,
  savePredictions = "final"
)

# ============================================================
# MODELO 1 — LPM baseline
# ============================================================
cat("\n>>> [lpm - 1/1] Linear Probability Model...\n")
tic("LPM baseline")
set.seed(SEED)

m1 <- train(
  pobre_num ~ .,
  data      = train |> select(-id, -pobre),
  method    = "lm",
  trControl = ctrl
)

opt1 <- optimizar_threshold(m1, train_lpm, train$pobre_num)
cat("    Threshold óptimo:     ", round(opt1$threshold, 4), "\n")
cat("    F1 (threshold óptimo):", round(opt1$f1, 4), "\n")

guardar_modelo(m1, "lpm_baseline", TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO)
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen LPM\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre)"))
rm(ctrl, dir_modelo, TIPO)
gc()