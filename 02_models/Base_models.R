# ============================================================
# base.R
# Modelos base — Logística baseline
# ============================================================

TIPO      <- "Base_models"
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
# MODELO 1 — Logística baseline
# ============================================================
cat("\n>>> [base - 1/1] Logística baseline...\n")
tic("Logística baseline")
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
toc()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre)"))
rm(ctrl, dir_modelo, TIPO)
gc()