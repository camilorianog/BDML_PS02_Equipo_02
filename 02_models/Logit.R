# ============================================================
# logit.R
# Modelos de Regresión Logística
# ============================================================

TIPO       <- "logit"
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
guardar_modelo(m1, "logit_baseline", TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO)

# ============================================================
# MODELO 2 — 
# ============================================================
cat("\n>>> [logit - 2/2] ...\n")
set.seed(SEED)


# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen logit\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(m1, m2, ctrl, opt1, opt2, dir_modelo, TIPO)
gc()