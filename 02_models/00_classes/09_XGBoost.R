# ============================================================
# 13_generate_submission_xgb_model13.R
# Genera la submission del XGBoost model_id 13
# usando xgb_push_to_70_summary_parcial.csv
# ============================================================

#XGBoost binario con nrounds=400, max_depth=6, eta=0.05, 
#min_child_weight, subsample, colsample_bytree y gamma ajustados; 
#threshold óptimo calibrado para maximizar F1.


# ------------------------------------------------------------
# 0. PAQUETES
# ------------------------------------------------------------
required_pkgs <- c("tidyverse", "xgboost", "readr")

installed <- rownames(installed.packages())
to_install <- setdiff(required_pkgs, installed)

if (length(to_install) > 0) {
  install.packages(to_install, dependencies = TRUE)
}

invisible(lapply(required_pkgs, library, character.only = TRUE))
options(menu.graphics = FALSE)

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------
SEED <- 123

dir_results <- "~/Library/CloudStorage/OneDrive-UniversidaddelosAndes/MAESTRÍA-PRIMER SEMESTRE/BIG DATA & MACHINE LEARNING/Problem Set 2 - Taller 2/BDML_PS02_Equipo_02-main/Results"
dir_results <- path.expand(dir_results)

summary_file <- file.path(dir_results, "xgb_push_to_70_summary_parcial.csv")

# model_id que quieres sacar
target_model_ids <- c(13, 8)

# ------------------------------------------------------------
# 2. CARGAR DATA
# ------------------------------------------------------------
train <- readRDS(file.path(dir_results, "train_model_ready.rds"))
test  <- readRDS(file.path(dir_results, "test_model_ready.rds"))

tabla_xgb <- readr::read_csv(summary_file, show_col_types = FALSE)

# ------------------------------------------------------------
# 3. VALIDAR QUE EXISTAN LOS MODEL_ID
# ------------------------------------------------------------
faltantes <- setdiff(target_model_ids, tabla_xgb$model_id)

if (length(faltantes) > 0) {
  stop(
    "Estos model_id no están en xgb_push_to_70_summary_parcial.csv: ",
    paste(faltantes, collapse = ", ")
  )
}

configs_objetivo <- tabla_xgb |>
  dplyr::filter(model_id %in% target_model_ids) |>
  dplyr::arrange(dplyr::desc(cv_f1))

print(configs_objetivo)

# ------------------------------------------------------------
# 4. PREPARAR MATRICES
# ------------------------------------------------------------
train_id <- train$id
test_id  <- test$id

y <- train$pobre

train_x <- train |> dplyr::select(-id, -pobre)
test_x  <- test  |> dplyr::select(-id)

X_train <- model.matrix(~ . - 1, data = train_x)
X_test  <- model.matrix(~ . - 1, data = test_x)

common_cols <- intersect(colnames(X_train), colnames(X_test))
X_train <- X_train[, common_cols, drop = FALSE]
X_test  <- X_test[, common_cols, drop = FALSE]

dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y)
dtest  <- xgboost::xgb.DMatrix(data = X_test)

# ------------------------------------------------------------
# 5. PESO DE CLASE
# ------------------------------------------------------------
n_pos <- sum(y == 1)
n_neg <- sum(y == 0)
scale_pos_weight_base <- n_neg / n_pos

cat("scale_pos_weight_base =", round(scale_pos_weight_base, 4), "\n")

# ------------------------------------------------------------
# 6. ENTRENAR Y GENERAR SUBMISSIONS
# ------------------------------------------------------------
for (i in seq_len(nrow(configs_objetivo))) {
  
  row_cfg <- configs_objetivo[i, ]
  
  cat("\n====================================================\n")
  cat("Entrenando model_id:", row_cfg$model_id, "\n")
  print(row_cfg)
  cat("====================================================\n")
  
  set.seed(SEED)
  
  final_xgb <- xgboost::xgb.train(
    params = list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = row_cfg$eta,
      max_depth = as.integer(row_cfg$max_depth),
      min_child_weight = row_cfg$min_child_weight,
      subsample = row_cfg$subsample,
      colsample_bytree = row_cfg$colsample_bytree,
      gamma = row_cfg$gamma,
      scale_pos_weight = scale_pos_weight_base,
      tree_method = "hist"
    ),
    data = dtrain,
    nrounds = as.integer(row_cfg$nrounds),
    verbose = 0
  )
  
  prob_test_xgb <- predict(final_xgb, dtest)
  pred_test_xgb <- ifelse(prob_test_xgb >= row_cfg$threshold, 1, 0)
  
  submission_xgb <- tibble::tibble(
    id = test_id,
    pobre = pred_test_xgb
  )
  
  out_name <- paste0(
    "submission_xgb_modelid_", row_cfg$model_id,
    "_nr", row_cfg$nrounds,
    "_dep", row_cfg$max_depth,
    "_eta", gsub("\\.", "", as.character(row_cfg$eta)),
    ".csv"
  )
  
  readr::write_csv(
    submission_xgb,
    file.path(dir_results, out_name)
  )
  
  model_rds_name <- gsub("\\.csv$", ".rds", out_name)
  saveRDS(final_xgb, file.path(dir_results, model_rds_name))
  
  cat("Guardado:", out_name, "\n")
}

# ------------------------------------------------------------
# 7. RESUMEN
# ------------------------------------------------------------
readr::write_csv(
  configs_objetivo,
  file.path(dir_results, "xgb_selected_models_13_8_summary.csv")
)

cat("\nListo. Se generaron las submissions para los model_id:\n")
cat(paste(target_model_ids, collapse = ", "), "\n")