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

# --- Limpiar entorno ----------------------------------------
rm(m1, m2, m3, m4, m5, ctrl,
   opt1, opt2, opt3, opt4, opt5,
   dir_dia, DIA)
gc()