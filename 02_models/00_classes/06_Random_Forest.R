# ============================================================
# Random_Forest.R
# Random Forest models — ranger directo
# ============================================================

TIPO       <- "06_Random_Forest"
dir_modelo <- here(paths$models, TIPO)
dir.create(dir_modelo, recursive = TRUE, showWarnings = FALSE)

# --- Cargar datos -------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

train <- train |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no_pobre", "pobre")))

# --- Parámetro base -----------------------------------------
p            <- ncol(train) - 2
mtry_default <- floor(sqrt(p))

# --- Helper: entrenar ranger + evaluar OOB F1 ---------------
train_ranger <- function(grid_row, importance = "none") {
  fit <- ranger(
    pobre         ~ .,
    data          = train |> select(-id),
    num.trees     = grid_row$num.trees,
    mtry          = grid_row$mtry,
    splitrule     = grid_row$splitrule,
    min.node.size = grid_row$min.node.size,
    probability   = TRUE,
    importance    = importance,
    num.threads   = parallel::detectCores() - 1,
    seed          = SEED
  )
  opt <- optimizar_threshold(fit, NULL, train$pobre)
  cat(sprintf("    num.trees=%d | mtry=%d | node=%d | splitrule=%s | OOB F1=%.4f\n",
              grid_row$num.trees, grid_row$mtry, grid_row$min.node.size,
              grid_row$splitrule, opt$f1))
  list(model = fit, opt = opt)
}

# --- Helper: seleccionar mejor modelo de grid ---------------
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

m1 <- ranger(
  pobre         ~ .,
  data          = train |> select(-id),
  num.trees     = 1000,
  mtry          = mtry_default,
  splitrule     = "gini",
  min.node.size = 1,
  probability   = TRUE,
  importance    = "permutation",
  num.threads   = parallel::detectCores() - 1,
  seed          = SEED
)

imp_df <- data.frame(
  variable   = names(m1$variable.importance),
  importance = m1$variable.importance
) |> arrange(desc(importance))

print(imp_df)

opt1      <- optimizar_threshold(m1, NULL, train$pobre)
nombre_m1 <- paste0("RF_mtry_", mtry_default)
guardar_modelo(m1, nombre_m1, TIPO, dir_modelo, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, TIPO, nombre_m1)
cat("    OOB Brier:", round(m1$prediction.error, 4), "\n")
toc()

# ============================================================
# MODELO 2 — RF grid completo (mtry, node size, splitrule)
# ============================================================
cat("\n>>> [rf - 2/4] RF grid completo...\n")
tic("RF grid completo")
set.seed(SEED)

grid_m2 <- expand.grid(
  num.trees     = 1000,
  mtry          = c(#floor(mtry_default / 2),
                    mtry_default,
                    floor(mtry_default * 1.5),
                    floor(mtry_default * 2),
                    p),
  min.node.size = c(1, 5, 10),
  splitrule     = c("hellinger", "extratrees"),
  stringsAsFactors = FALSE
)

results_m2 <- map(seq_len(nrow(grid_m2)), ~ train_ranger(grid_m2[.x, ]))
best2      <- best_from_grid(results_m2)
nombre_m2  <- paste0("RF_",       best2$model$splitrule,
                     "_mtry_",    best2$model$mtry,
                     "_node_",    best2$model$min.node.size,
                     "_trees_",   best2$model$num.trees)
guardar_modelo(best2$model, nombre_m2, TIPO, dir_modelo,
               best2$opt$threshold, best2$opt$f1)
generar_submission(best2$model, test, best2$opt$threshold, TIPO, nombre_m2)
toc()

# ============================================================
# MODELO 3 — RF Low Var
# ============================================================
cat("\n>>> [rf - 3/4] RF Low Var...\n")
tic("RF Low Var")
set.seed(SEED)

m3 <- ranger(
  pobre         ~ .,
  data          = train |> select(-id),
  num.trees     = 1000,
  mtry          = mtry_default,
  splitrule     = "extratrees",
  min.node.size = 10,
  probability   = TRUE,
  importance    = "permutation",
  num.threads   = parallel::detectCores() - 1,
  seed          = SEED
)

imp_df <- data.frame(
  variable   = names(m3$variable.importance),
  importance = m3$variable.importance
) |> arrange(desc(importance))

print(imp_df)

opt3      <- optimizar_threshold(m3, NULL, train$pobre)
nombre_m3 <- paste0("RF_Low_Var")
guardar_modelo(m3, nombre_m3, TIPO, dir_modelo, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, TIPO, nombre_m3)
cat("    OOB Brier:", round(m3$prediction.error, 4), "\n")
toc()

# ============================================================
# MODELO 4 — RF Hellinger+Bagging
# ============================================================
cat("\n>>> [rf - 3/3] RF Hellinger+Bagging...\n")
tic("RF Hellinger+Bagging")
set.seed(SEED)

m4 <- ranger(
  pobre         ~ .,
  data          = train |> select(-id),
  num.trees     = 1000,
  mtry          = p,
  splitrule     = "hellinger",
  min.node.size = 10,
  probability   = TRUE,
  #importance    = "permutation",
  num.threads   = parallel::detectCores() - 1,
  seed          = SEED
)

#imp_df <- data.frame(
 # variable   = names(m4$variable.importance),
#  importance = m4$variable.importance
#) |> arrange(desc(importance))

#print(imp_df)

opt4      <- optimizar_threshold(m4, NULL, train$pobre)
nombre_m4 <- paste0("RF_HD_Bagging")
guardar_modelo(m4, nombre_m4, TIPO, dir_modelo, opt4$threshold, opt4$f1)
generar_submission(m4, test, opt4$threshold, TIPO, nombre_m4)
cat("    OOB Brier:", round(m4$prediction.error, 4), "\n")
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
rm(list = ls(pattern = "^(m[0-9]+|opt[0-9]+|nombre|best|grid|results|fit)"))
rm(dir_modelo, TIPO, p, mtry_default, train_ranger, best_from_grid)
gc()
