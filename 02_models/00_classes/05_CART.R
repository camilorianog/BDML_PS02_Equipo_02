# ============================================================
# 05_CART.R
# Classification and Regression Trees
# ============================================================

TIPO       <- "05_CART"
dir_modelo <- here(paths$submissions, TIPO)
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
# MODELO 1 — CART default
# ============================================================
cat("\n>>> [cart - 1/4] CART default...\n")
tic("CART default")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rpart",
  trControl = ctrl,
  metric    = "AUC"
)

opt1      <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- paste0("CART_cp_", format(round(m1$bestTune$cp, 6), scientific = FALSE))
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO, nombre_m1)
toc()

# ============================================================
# MODELO 2 — CART grid cp fino
# ============================================================
cat("\n>>> [cart - 2/4] CART grid cp fino...\n")
tic("CART grid cp fino")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rpart",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(cp = c(0, 10^seq(-6, -4, length = 10)))
)

opt2      <- optimizar_threshold(m2, train, train$pobre)
nombre_m2 <- paste0("CART_cp_", format(round(m2$bestTune$cp, 6), scientific = FALSE))
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# MODELO 3 — CART grid cp amplio
# ============================================================
cat("\n>>> [cart - 3/4] CART grid cp amplio...\n")
tic("CART grid cp amplio")
set.seed(SEED)

m3 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rpart",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(cp = 10^seq(-4, -1, length = 30))
)

opt3      <- optimizar_threshold(m3, train, train$pobre)
nombre_m3 <- paste0("CART_cp_", format(round(m3$bestTune$cp, 6), scientific = FALSE))
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, TIPO, nombre_m3)
toc()

# ============================================================
# MODELO 4 — CART podado (anti-overfitting)
# cp en rango conservador + restricciones de nodo
# ============================================================
cat("\n>>> [cart - 4/4] CART podado...\n")
tic("CART podado")
set.seed(SEED)

m4 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rpart",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid = expand.grid(cp = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05)),
  control = rpart::rpart.control(
    minsplit  = 30,
    minbucket = 10,
    maxdepth  = 10
  )
)

opt4      <- optimizar_threshold(m4, train, train$pobre)
nombre_m4 <- paste0("CART_podado_cp_",
                    format(round(m4$bestTune$cp, 6), scientific = FALSE))
guardar_modelo(m4, nombre_m4, TIPO, dir_modelo, opt4$threshold, opt4$f1)
generar_submission(m4, test, opt4$threshold, TIPO, nombre_m4)
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen CART\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre)"))
rm(ctrl, dir_modelo, TIPO)
gc()
