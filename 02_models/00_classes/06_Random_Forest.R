# ============================================================
# 06_Random_Forest.R
# Random Forest — ranger directo
#
# Estrategia (según project instructions):
#   - ntree = 100 (165k obs → 1000 es prohibitivo)
#   - Threshold se escoge en VALIDATION SET (split 80/20),
#     NO con OOB del training completo (infla F1 artificialmente).
#   - Modelo final se refitea en el 100% del train con el
#     threshold ya seleccionado.
# ============================================================

TIPO       <- "06_Random_Forest"
dir_modelo <- here(paths$submissions, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

# --- Cargar datos -------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

train <- train |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# --- Parámetros base ----------------------------------------
NTREES       <- 100
p            <- ncol(train) - 2            # -id -pobre
mtry_default <- floor(sqrt(p))
n_threads    <- max(1, parallel::detectCores() - 1)

# --- Split 80/20 para threshold tuning ----------------------
set.seed(SEED)
idx_tr    <- caret::createDataPartition(train$pobre, p = 0.8, list = FALSE)
train_tr  <- train[idx_tr,  ]
train_val <- train[-idx_tr, ]

# --- Helper: buscar threshold óptimo en validation set ------
f1_threshold_search <- function(probs, target_bin,
                                thresholds = seq(0.20, 0.60, by = 0.005)) {
  f1s <- vapply(thresholds, function(t) {
    preds <- as.integer(probs >= t)
    tp <- sum(preds == 1 & target_bin == 1)
    fp <- sum(preds == 1 & target_bin == 0)
    fn <- sum(preds == 0 & target_bin == 1)
    pr <- if (tp + fp == 0) 0 else tp / (tp + fp)
    rc <- if (tp + fn == 0) 0 else tp / (tp + fn)
    if (pr + rc == 0) 0 else 2 * pr * rc / (pr + rc)
  }, numeric(1))
  list(threshold = thresholds[which.max(f1s)],
       f1        = max(f1s, na.rm = TRUE))
}

# --- Helper: fit en 80%, eval en 20%, refit en 100% ---------
fit_rf <- function(mtry, min.node.size, splitrule,
                   num.trees = NTREES, importance = "none",
                   etiqueta = "") {
  # Fit en train_tr (80%)
  fit_tr <- ranger(
    pobre         ~ .,
    data          = train_tr |> select(-id),
    num.trees     = num.trees,
    mtry          = mtry,
    splitrule     = splitrule,
    min.node.size = min.node.size,
    probability   = TRUE,
    num.threads   = n_threads,
    seed          = SEED
  )

  # Predecir en train_val (20%) y optimizar threshold
  probs_val  <- predict(fit_tr, data = train_val |> select(-id))$predictions[, "pobre"]
  target_val <- as.integer(train_val$pobre == "pobre")
  opt        <- f1_threshold_search(probs_val, target_val)

  cat(sprintf("    %s | mtry=%d | node=%d | split=%s | val F1=%.4f | t=%.3f\n",
              etiqueta, mtry, min.node.size, splitrule, opt$f1, opt$threshold))

  # Refit en 100% del train con mismos hiperparámetros
  fit_full <- ranger(
    pobre         ~ .,
    data          = train |> select(-id),
    num.trees     = num.trees,
    mtry          = mtry,
    splitrule     = splitrule,
    min.node.size = min.node.size,
    probability   = TRUE,
    importance    = importance,
    num.threads   = n_threads,
    seed          = SEED
  )

  list(model = fit_full, opt = opt,
       mtry = mtry, min.node.size = min.node.size,
       splitrule = splitrule, num.trees = num.trees)
}

best_from_grid <- function(results) {
  f1s <- map_dbl(results, ~ .x$opt$f1)
  results[[which.max(f1s)]]
}

# ============================================================
# MODELO 1 — RF mtry default
# ============================================================
cat("\n>>> [rf - 1/4] Random Forest mtry default...\n")
tic("RF mtry default")
set.seed(SEED)

best1 <- fit_rf(mtry = mtry_default, min.node.size = 1, splitrule = "gini",
                importance = "permutation", etiqueta = "m1 default")

imp_df <- data.frame(
  variable   = names(best1$model$variable.importance),
  importance = best1$model$variable.importance
) |> arrange(desc(importance))
print(imp_df)

nombre_m1 <- paste0("RF_mtry_", mtry_default)
guardar_modelo(best1$model, nombre_m1, TIPO, dir_modelo,
               best1$opt$threshold, best1$opt$f1)
generar_submission(best1$model, test, best1$opt$threshold, TIPO, nombre_m1)
cat("    OOB Brier:", round(best1$model$prediction.error, 4), "\n")
toc()

# ============================================================
# MODELO 2 — RF grid (mtry × min.node.size × splitrule)
# ============================================================
cat("\n>>> [rf - 2/4] RF grid completo...\n")
tic("RF grid completo")
set.seed(SEED)

grid_m2 <- expand.grid(
  mtry          = c(mtry_default,
                    floor(mtry_default * 1.5),
                    floor(mtry_default * 2),
                    p),
  min.node.size = c(1, 5, 10),
  splitrule     = c("gini", "hellinger", "extratrees"),
  stringsAsFactors = FALSE
)

results_m2 <- map(seq_len(nrow(grid_m2)), function(i) {
  fit_rf(mtry          = grid_m2$mtry[i],
         min.node.size = grid_m2$min.node.size[i],
         splitrule     = grid_m2$splitrule[i],
         etiqueta      = sprintf("grid[%02d/%d]", i, nrow(grid_m2)))
})
best2 <- best_from_grid(results_m2)

nombre_m2 <- paste0("RF_",      best2$splitrule,
                    "_mtry_",   best2$mtry,
                    "_node_",   best2$min.node.size,
                    "_trees_",  best2$num.trees)
guardar_modelo(best2$model, nombre_m2, TIPO, dir_modelo,
               best2$opt$threshold, best2$opt$f1)
generar_submission(best2$model, test, best2$opt$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# MODELO 3 — RF Low Var (extratrees + node=10)
# ============================================================
cat("\n>>> [rf - 3/4] RF Low Var...\n")
tic("RF Low Var")
set.seed(SEED)

best3 <- fit_rf(mtry = mtry_default, min.node.size = 10,
                splitrule = "extratrees",
                importance = "permutation", etiqueta = "m3 low-var")

imp_df <- data.frame(
  variable   = names(best3$model$variable.importance),
  importance = best3$model$variable.importance
) |> arrange(desc(importance))
print(imp_df)

nombre_m3 <- "RF_Low_Var"
guardar_modelo(best3$model, nombre_m3, TIPO, dir_modelo,
               best3$opt$threshold, best3$opt$f1)
generar_submission(best3$model, test, best3$opt$threshold, TIPO, nombre_m3)
cat("    OOB Brier:", round(best3$model$prediction.error, 4), "\n")
toc()

# ============================================================
# MODELO 4 — RF Hellinger + Bagging (mtry = p)
# ============================================================
cat("\n>>> [rf - 4/4] RF Hellinger + Bagging...\n")
tic("RF Hellinger + Bagging")
set.seed(SEED)

best4 <- fit_rf(mtry = p, min.node.size = 10,
                splitrule = "hellinger", etiqueta = "m4 hd-bag")

nombre_m4 <- "RF_HD_Bagging"
guardar_modelo(best4$model, nombre_m4, TIPO, dir_modelo,
               best4$opt$threshold, best4$opt$f1)
generar_submission(best4$model, test, best4$opt$threshold, TIPO, nombre_m4)
cat("    OOB Brier:", round(best4$model$prediction.error, 4), "\n")
toc()

# ============================================================
# RESUMEN
# ============================================================
cat("\n======================================================\n")
cat("  Resumen Random Forest\n")
cat("======================================================\n")
read.csv(here(paths$models, "log.csv")) |>
  dplyr::filter(tipo == TIPO) |>
  arrange(desc(cv_f1)) |>
  print()

# --- Limpiar entorno ----------------------------------------
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre|best|grid|results|fit|imp)"))
rm(dir_modelo, TIPO, p, mtry_default, n_threads, NTREES,
   train_tr, train_val, idx_tr,
   fit_rf, best_from_grid, f1_threshold_search)
gc()
