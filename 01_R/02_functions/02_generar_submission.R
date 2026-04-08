# ============================================================
# 02_generar_submission.R
# Genera el CSV de submission para Kaggle
# ============================================================

generar_submission <- function(modelo, test, threshold, dia, nombre = NULL) {
  
  # Si no se pasa nombre, construirlo desde el método
  if (is.null(nombre)) {
    metodo <- modelo$method
    if (!is.null(modelo$bestTune) && ncol(modelo$bestTune) > 0) {
      params <- paste(names(modelo$bestTune),
                      unlist(modelo$bestTune),
                      sep = "_", collapse = "_")
      nombre <- paste0(metodo, "_", params)
    } else {
      nombre <- metodo
    }
  }
  
  probs <- tryCatch(
    predict(modelo, test, type = "prob")[, "1"],
    error = function(e) {
      tryCatch(
        predict(modelo, test, type = "prob")[, "pobre"],
        error = function(e2) {
          warning("No se pudo obtener probabilidades para '",
                  modelo$method, "'. Usando predicción de clase directa.")
          as.integer(predict(modelo, test) == 1)
        }
      )
    }
  )
  
  preds <- if (is.numeric(probs)) as.integer(probs >= threshold) else probs
  
  submission <- data.frame(id = test$id, pobre = preds)
  
  dir_sub <- here(paths$submissions, paste0(dia, "_day"))
  dir.create(dir_sub, recursive = TRUE, showWarnings = FALSE)
  
  ruta <- file.path(dir_sub, paste0(nombre, ".csv"))
  write.csv(submission, ruta, row.names = FALSE)
  
  cat("    Submission guardada:", basename(ruta), "\n")
}