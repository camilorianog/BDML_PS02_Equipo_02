# ============================================================
# 02_generar_submission.R
# ============================================================
generar_submission <- function(modelo, test, threshold, tipo, nombre = NULL) {
  
  if (is.null(nombre)) {
    metodo <- modelo$method
    if (!is.null(modelo$bestTune) && ncol(modelo$bestTune) > 0) {
      params <- paste(names(modelo$bestTune),
                      format(unlist(modelo$bestTune), scientific = FALSE),
                      sep = "_", collapse = "_")
      nombre <- paste0(metodo, "_", params)
    } else {
      nombre <- metodo
    }
  }
  
  probs <- tryCatch(
    if ("ranger" %in% class(modelo)) {
      predict(modelo, test)$predictions[, "pobre"]
    } else {
      predict(modelo, test, type = "prob")[, "pobre"]
    },
    error = function(e) {
      tryCatch(
        predict(modelo, test, type = "prob")[, "1"],
        error = function(e2) {
          warning("No se pudo obtener probabilidades para '",
                  modelo$method, "'. Usando predicción de clase directa.")
          if ("ranger" %in% class(modelo)) {
            as.numeric(predict(modelo, test)$predictions[, "pobre"])
          } else {
            as.numeric(predict(modelo, test))
          }
        }
      )
    }
  )
  
  preds <- as.integer(probs >= threshold)
  submission <- data.frame(id = test$id, pobre = preds)
  
  dir_sub <- file.path(paths$submissions, tipo)         
  dir.create(dir_sub, recursive = TRUE, showWarnings = FALSE)
  
  ruta <- file.path(dir_sub, paste0(nombre, ".csv"))
  write.csv(submission, ruta, row.names = FALSE)
  
  cat("    Submission guardada:", basename(ruta), "\n")
}