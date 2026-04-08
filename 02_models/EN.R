# ============================================================
# EN.R — Modelos Elastic Net
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
  number          = CV_FOLDS,
  classProbs      = TRUE,
  summaryFunction = prSummary,
  savePredictions = "final"
)


# ============================================================
# DÍA 00
# ============================================================

DIA     <- "00"
dir_dia <- here(paths$models, paste0(DIA, "_day"))
dir.create(dir_dia, recursive = TRUE, showWarnings = FALSE)

# --- MODELO 1 — Logística baseline -------------------------
cat("\n>>> [1] Logística baseline...\n")
set.seed(SEED)
start_time <- Sys.time()

m1 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glm",
  family    = binomial(link = "logit"),
  trControl = ctrl,
  metric    = "AUC"
)

opt1     <- optimizar_threshold(m1, train, train$pobre)
nombre_m1 <- "logit_baseline"

guardar_modelo(m1, nombre_m1, DIA, dir_dia, opt1$threshold, opt1$f1)
generar_submission(m1, test, opt1$threshold, DIA, nombre_m1)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))


# ============================================================
# DÍA 01
# ============================================================

DIA     <- "01"
dir_dia <- here(paths$models, paste0(DIA, "_day"))
dir.create(dir_dia, recursive = TRUE, showWarnings = FALSE)

# --- MODELO 2 — Ridge (alpha = 0) --------------------------
cat("\n>>> [2] Ridge (alpha = 0)...\n")
set.seed(SEED)
start_time <- Sys.time()

m2 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    alpha  = 0,
    lambda = 10^seq(-4, 1, length = 20)
  )
)

opt2      <- optimizar_threshold(m2, train, train$pobre) 
nombre_m2 <- paste0(
  "EN_lambda_", format(round(m2$bestTune$lambda, 6), scientific = FALSE),
  "_alpha_",    m2$bestTune$alpha
)

guardar_modelo(m2, nombre_m2, DIA, dir_dia, opt2$threshold, opt2$f1)
generar_submission(m2, test, opt2$threshold, DIA, nombre_m2)  

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# --- MODELO 3 — Lasso (alpha = 1) --------------------------
cat("\n>>> [3] Lasso (alpha = 1)...\n")
set.seed(SEED)
start_time <- Sys.time()

m3 <- train(                                               
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    alpha  = 1,
    lambda = 10^seq(-4, 1, length = 20)
  )
)

opt3      <- optimizar_threshold(m3, train, train$pobre)
nombre_m3 <- paste0(
  "EN_lambda_", format(round(m3$bestTune$lambda, 6), scientific = FALSE),
  "_alpha_",    m3$bestTune$alpha
)

guardar_modelo(m3, nombre_m3, DIA, dir_dia, opt3$threshold, opt3$f1)
generar_submission(m3, test, opt3$threshold, DIA, nombre_m3)  

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# --- MODELO 4 — Elastic Net mix (alpha = 0.5) --------------
cat("\n>>> [4] Elastic Net mix (alpha = 0.5)...\n")        
set.seed(SEED)
start_time <- Sys.time()

m4 <- train(                                               
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = expand.grid(
    alpha  = 0.5,
    lambda = 10^seq(-4, 1, length = 20)
  )
)

opt4      <- optimizar_threshold(m4, train, train$pobre)  
nombre_m4 <- paste0(
  "EN_lambda_", format(round(m4$bestTune$lambda, 6), scientific = FALSE),
  "_alpha_",    m4$bestTune$alpha
)

guardar_modelo(m4, nombre_m4, DIA, dir_dia, opt4$threshold, opt4$f1)
generar_submission(m4, test, opt4$threshold, DIA, nombre_m4)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))


# ============================================================
# DÍA 03
# ============================================================

DIA     <- "03"
dir_dia <- here(paths$models, paste0(DIA, "_day"))
dir.create(dir_dia, recursive = TRUE, showWarnings = FALSE)

# --- MODELO 5 — EN full grid, metric AUC -------------------
cat("\n>>> [5] EN full grid (AUC)...\n")                   
set.seed(SEED)
start_time <- Sys.time()

m5 <- train(
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "AUC",
  tuneGrid  = EN_GRID
)

opt5      <- optimizar_threshold(m5, train, train$pobre)
nombre_m5 <- paste0(
  "EN_lambda_", format(round(m5$bestTune$lambda, 6), scientific = FALSE),
  "_alpha_",    m5$bestTune$alpha
)

guardar_modelo(m5, nombre_m5, DIA, dir_dia, opt5$threshold, opt5$f1)
generar_submission(m5, test, opt5$threshold, DIA, nombre_m5)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

# --- MODELO 6 — EN full grid, metric F ---------------------
cat("\n>>> [6] EN full grid (F)...\n")
set.seed(SEED)
start_time <- Sys.time()

m6 <- train(                                               
  pobre ~ .,
  data      = train |> select(-id),
  method    = "glmnet",
  family    = "binomial",
  trControl = ctrl,
  metric    = "F",                                        
  tuneGrid  = EN_GRID
)

opt6      <- optimizar_threshold(m6, train, train$pobre)
nombre_m6 <- paste0(
  "EN_lambda_", format(round(m6$bestTune$lambda, 6), scientific = FALSE),
  "_alpha_",    m6$bestTune$alpha
)

guardar_modelo(m6, nombre_m6, DIA, dir_dia, opt6$threshold, opt6$f1)
generar_submission(m6, test, opt6$threshold, DIA, nombre_m6)

cat(sprintf("Tiempo: %.1f min\n", as.numeric(difftime(Sys.time(), start_time, units = "mins"))))
