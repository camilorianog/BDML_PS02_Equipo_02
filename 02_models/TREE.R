# ============================================================
# TREE.R — Modelos de Árbol: CART y Random Forest
# ============================================================

# --- Cargar datos -------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

train <- train |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# --- Control de entrenamiento -------------------------------
ctrl <- trainControl(
  method          = "cv",
  number          = CV_FOLDS,  # definido en 00_run.R
  classProbs      = TRUE,
  summaryFunction = prSummary,
  savePredictions = "final"
)

# ============================================================
# DÍA 04 — CART y RF
# ============================================================

DIA     <- "04"
dir_dia <- here(paths$models, paste0(DIA, "_day"))
dir.create(dir_dia, recursive = TRUE, showWarnings = FALSE)

# --- MODELO 7 — CART default --------------------------------
cat("\n>>> [7] CART default...\n")
set.seed(SEED)
start_time <- Sys.time()

m7 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rpart",
  trControl = ctrl,
  metric    = "AUC"
)

opt7      <- optimizar_threshold(m7, train, train$pobre)
nombre_m7 <- paste0("CART_cp_", format(round(m7$bestTune$cp, 6), scientific = FALSE))

guardar_modelo(m7, nombre_m7, DIA, dir_dia, opt7$threshold, opt7$f1)
generar_submission(m7, test, opt7$threshold, DIA, nombre_m7)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# --- MODELO 8 — CART grid cp --------------------------------
cat("\n>>> [8] CART grid cp...\n")
set.seed(SEED)
start_time <- Sys.time()

m8 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rpart",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(cp = c(0, 10^seq(-6, -4, length = 10)))
)

opt8      <- optimizar_threshold(m8, train, train$pobre)
nombre_m8 <- paste0("CART_cp_", format(round(m8$bestTune$cp, 6), scientific = FALSE))

guardar_modelo(m8, nombre_m8, DIA, dir_dia, opt8$threshold, opt8$f1)
generar_submission(m8, test, opt8$threshold, DIA, nombre_m8)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# --- MODELO 9 — CART grid cp amplio -------------------------
cat("\n>>> [9] CART grid cp amplio...\n")
set.seed(SEED)
start_time <- Sys.time()

m9 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rpart",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(cp = 10^seq(-4, -1, length = 30))
)

opt9      <- optimizar_threshold(m9, train, train$pobre)
nombre_m9 <- paste0("CART_cp_", format(round(m9$bestTune$cp, 6), scientific = FALSE))

guardar_modelo(m9, nombre_m9, DIA, dir_dia, opt9$threshold, opt9$f1)
generar_submission(m9, test, opt9$threshold, DIA, nombre_m9)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# --- MODELO 10 — RF mtry default ----------------------------
cat("\n>>> [10] Random Forest mtry default...\n")
set.seed(SEED)
start_time <- Sys.time()

p            <- ncol(train) - 2  # descontar id y pobre
mtry_default <- floor(sqrt(p))

m10 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rf",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(mtry = mtry_default),
  ntree     = 100
)

preds_m10  <- factor(
  ifelse(predict(m10, train, type = "prob")[, "pobre"] >= 0.5, "pobre", "no_pobre"),
  levels = c("no_pobre", "pobre")
)

f1_m10     <- confusionMatrix(preds_m10, train$pobre, positive = "pobre")$byClass["F1"]
opt10      <- list(threshold = 0.5, f1 = f1_m10)
nombre_m10 <- paste0("RF_mtry_", m10$bestTune$mtry)

guardar_modelo(m10, nombre_m10, DIA, dir_dia, opt10$threshold, opt10$f1)
generar_submission(m10, test, opt10$threshold, DIA, nombre_m10)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# --- MODELO 11 — RF grid reducido ---------------------------
cat("\n>>> [11] Random Forest grid reducido...\n")
set.seed(SEED)
start_time <- Sys.time()

m11 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "rf",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    mtry = c(floor(mtry_default / 2), mtry_default, floor(mtry_default * 2))
  ),
  ntree     = 100
)

preds_m11  <- factor(
  ifelse(predict(m11, train, type = "prob")[, "pobre"] >= 0.5, "pobre", "no_pobre"),
  levels = c("no_pobre", "pobre")
)

f1_m11     <- confusionMatrix(preds_m11, train$pobre, positive = "pobre")$byClass["F1"]
opt11      <- list(threshold = 0.5, f1 = f1_m11)
nombre_m11 <- paste0("RF_mtry_", m11$bestTune$mtry)

guardar_modelo(m11, nombre_m11, DIA, dir_dia, opt11$threshold, opt11$f1)
generar_submission(m11, test, opt11$threshold, DIA, nombre_m11)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# ============================================================
# DÍA 05 — Boosting
# ============================================================

DIA     <- "05"
dir_dia <- here(paths$models, paste0(DIA, "_day"))
dir.create(dir_dia, recursive = TRUE, showWarnings = FALSE)