# ============================================================
# 01_guardar_modelo.R
# Guarda el modelo .rds y registra resultados en log.csv
# ============================================================

guardar_modelo <- function(modelo, nombre, tipo, dir_modelo, threshold, f1_cv) {
  
  # Guardar .rds
  ruta_rds <- file.path(dir_modelo, paste0(nombre, ".rds"))
  saveRDS(modelo, ruta_rds)
  
  # Actualizar log
  log_entry <- data.frame(
    tipo      = tipo,
    nombre    = nombre,
    threshold = round(threshold, 3),
    cv_f1     = round(f1_cv, 4),
    kaggle_f1 = NA,
    stringsAsFactors = FALSE
  )
  
  ruta_log <- here(paths$models, "log.csv")
  if (file.exists(ruta_log)) {
    log_actual <- read.csv(ruta_log)
    log_actual <- log_actual |>
      filter(!(tipo == log_entry$tipo & nombre == log_entry$nombre))
    write.csv(rbind(log_actual, log_entry), ruta_log, row.names = FALSE)
  } else {
    write.csv(log_entry, ruta_log, row.names = FALSE)
  }
  
  cat("    Modelo guardado:", nombre, "\n")
  cat("    CV F1:", round(f1_cv, 4),
      "| Threshold:", round(threshold, 3), "\n")
}