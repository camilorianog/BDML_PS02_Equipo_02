# ============================================================
# 02_LPM.R
# Modelos de Probabilidad Lineal (LPM)
# ============================================================
TIPO       <- "02_LPM"
dir_modelo <- here(paths$submissions, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

# --- Cargar datos -------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

# LPM usa variable numérica (0/1), no factor
train <- train |>
  mutate(pobre_num = as.numeric(pobre))

# --- Features -----------------------------------------------
features <- setdiff(names(train), c("id", "pobre", "pobre_num"))

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
  x         = train |> select(all_of(features)),
  y         = train$pobre_num,
  method    = "lm",
  trControl = ctrl
)

opt1 <- optimizar_threshold(m1, train |> select(all_of(features)), train$pobre_num)
cat("    Threshold óptimo:     ", round(opt1$threshold, 4), "\n")
cat("    F1 (threshold óptimo):", round(opt1$f1, 4), "\n")
guardar_modelo(m1, "lpm_baseline", TIPO, dir_modelo, opt1$threshold, opt1$f1)

probs_lpm <- predict(m1, newdata = test |> select(all_of(features)))
probs_lpm <- pmin(pmax(probs_lpm, 0), 1)
preds_lpm <- as.integer(probs_lpm >= opt1$threshold)

submission <- data.frame(id = test$id, pobre = preds_lpm)
dir_sub    <- file.path(paths$submissions, TIPO)
dir.create(dir_sub, recursive = TRUE, showWarnings = FALSE)
write.csv(submission, file.path(dir_sub, "lpm_baseline.csv"), row.names = FALSE)
cat("    Submission guardada: lpm_baseline.csv\n")
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
rm(ctrl, dir_modelo, TIPO, features)
gc()