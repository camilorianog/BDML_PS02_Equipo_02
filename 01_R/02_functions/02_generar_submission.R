# ============================================================
# 02_generar_submission.R
# Genera el CSV de submission para Kaggle
# ============================================================

generar_submission <- function(modelo, test, threshold, dia) {
  
  # Extraer nombre desde el modelo
  metodo <- modelo$method
  
  if (metodo == "glm") {
    nombre_archivo <- "logit_baseline"
  } else if (metodo == "glmnet") {
    best <- modelo$bestTune
    alpha_str  <- gsub("\\.", "", sprintf("%.2f", best$alpha))
    lambda_str <- round(best$lambda, 4)
    nombre_archivo <- paste0("EN_lambda_", lambda_str,
                             "_alpha_", alpha_str)
  } else {
    nombre_archivo <- metodo
  }
  
  probs <- predict(modelo, test, type = "prob")[, "pobre"]
  preds <- as.integer(probs >= threshold)
  
  submission <- data.frame(
    id    = test$id,
    pobre = preds
  )
  
  dir_sub <- here(paths$submissions, paste0( dia, "_day"))
  dir.create(dir_sub, recursive = TRUE, showWarnings = FALSE)
  
  ruta <- file.path(dir_sub, paste0(nombre_archivo, ".csv"))
  write.csv(submission, ruta, row.names = FALSE)
  
  cat("    Submission guardada:", basename(ruta), "\n")
}