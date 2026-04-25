# ============================================================
# 07_Boosting.R
# Gradient Boosting models (XGBoost + LightGBM)
# ============================================================

TIPO       <- "07_Boosting"
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
# MODELO 1 — XGBoost default
# ============================================================
cat("\n>>> [boosting - 1/3] XGBoost default...\n")
tic("XGBoost default")
set.seed(SEED)

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "xgbTree",
  trControl = ctrl,
  metric    = "AUC"
)

opt1      <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- paste0("XGB_depth_", m1$bestTune$max_depth,
                    "_eta_",      m1$bestTune$eta)
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO, nombre_m1)
toc()

# ============================================================
# MODELO 2 — XGBoost grid reducido
# ============================================================
cat("\n>>> [boosting - 2/3] XGBoost grid reducido...\n")
tic("XGBoost grid reducido")
set.seed(SEED)

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "xgbTree",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    nrounds          = c(100, 300),
    max_depth        = c(3, 6),
    eta              = c(0.01, 0.1),
    gamma            = 0,
    colsample_bytree = 0.7,
    min_child_weight = 1,
    subsample        = 0.8
  )
)

opt2      <- optimizar_threshold(m2, train, train$pobre)
nombre_m2 <- paste0("XGB_depth_",  m2$bestTune$max_depth,
                    "_eta_",       m2$bestTune$eta,
                    "_rounds_",    m2$bestTune$nrounds)
guardar_modelo(m2, nombre_m2, TIPO, dir_modelo, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# Modelo 3 XGBoost Non Caret
# ============================================================

# --- Preparar matrices --------------------------------------
dummy_recipe <- dummyVars(~ ., data = train |> select(-id, -pobre),
                          fullRank = TRUE)

X_train <- predict(dummy_recipe, train |> select(-id, -pobre))
y_train <- as.numeric(train$pobre == "pobre")
X_test  <- predict(dummy_recipe, test |> select(-id))

dtrain  <- xgb.DMatrix(data = X_train, label = y_train)
dtest   <- xgb.DMatrix(data = X_test)

# --- Parámetros ---------------------------------------------
params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "auc",
  eta              = 0.05,
  max_depth        = 6,
  gamma            = 0,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  nthread          = parallel::detectCores() - 1
)

cat("\n>>> [xgb - 3/3] XGBoost nrounds=1000...\n")
tic("XGBoost")
nombre_m3 <- "xgb_depth6_eta005_rounds1000"

set.seed(SEED)
m3 <- xgb.train(
  params        = params,
  data          = dtrain,
  nrounds       = 1000,
  verbose       = 1,
  print_every_n = 100
)

# Guardar predicciones in-sample (para métricas deck1)
probs_train_m3 <- predict(m3, dtrain)
saveRDS(probs_train_m3,
        file.path(dir_modelo, paste0(nombre_m3, "_train_preds.rds")))

opt3 <- optimizar_threshold(m3, train, train$pobre)
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)

# --- Submission ---------------------------------------------
dir_sub <- here(paths$submissions, TIPO)
dir.create(dir_sub, recursive = TRUE, showWarnings = FALSE)
probs_test <- predict(m3, dtest)
preds_test <- as.integer(probs_test >= opt3$threshold)
sub        <- data.frame(id = test$id, pobre = preds_test)
write.csv(sub, file.path(dir_sub, paste0(nombre_m3, ".csv")), row.names = FALSE)
cat("    Submission guardada:", nombre_m3, ".csv\n")
toc()

# --- Importancia de variables -------------------------------
imp_xgb <- xgb.importance(feature_names = colnames(X_train), model = m3)
print(imp_xgb)
xgb.plot.importance(imp_xgb, top_n = 25)

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen Boosting\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre_m[0-9]+|probs_train)"))
rm(ctrl, dir_modelo, TIPO, dummy_recipe, X_train, X_test, y_train,
   dtrain, dtest, params, imp_xgb)
gc()