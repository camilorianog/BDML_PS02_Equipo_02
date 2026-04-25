# ============================================================
# 00_deck1_metrics.R
# Métricas y figuras para deck 1 (deep-dive del mejor modelo)
# Selección interactiva del modelo desde log.csv.
# Outputs:
#   04_outputs/tables/deck1_*.csv
#   04_outputs/figures/deck1_*.png
#
# Pre-requisitos: ejecutar antes 00_rundirectory.R (al menos la
# sección que define `paths`, `SEED` y carga paquetes), o
# correr este script desde un entorno donde esos objetos existan.
# ============================================================

if (!exists("paths")) {
  stop("paths no está definido. Corre primero 00_rundirectory.R o ",
       "carga manualmente: pacman::p_load(here, tidyverse, yardstick, ",
       "scales, caret, ranger) y define paths.")
}

pacman::p_load(here, tidyverse, yardstick, scales, caret, ranger)

# ------------------------------------------------------------
# 1. SELECCIÓN DEL MEJOR MODELO 
# ------------------------------------------------------------

ruta_log <- here(paths$models, "log.csv")
if (!file.exists(ruta_log)) {
  stop("No existe ", ruta_log, ". Entrena algún modelo primero.")
}

log_modelos <- read.csv(ruta_log, stringsAsFactors = FALSE) |>
  mutate(
    cv_f1     = suppressWarnings(as.numeric(cv_f1)),
    kaggle_f1 = suppressWarnings(as.numeric(kaggle_f1)),
    threshold = suppressWarnings(as.numeric(threshold))
  )

cat("\n================================================\n")
cat("  Modelos disponibles en log.csv\n")
cat("================================================\n")
print(log_modelos |> select(any_of(c("fila", "tipo", "nombre", "cv_f1", "kaggle_f1", "threshold"))) |>
        tibble::rownames_to_column("fila"), row.names = FALSE)

fila_sel <- suppressWarnings(
  as.integer(readline(prompt = "\nIngresa el número de fila del modelo a usar: "))
)
if (is.na(fila_sel) || fila_sel < 1 || fila_sel > nrow(log_modelos)) {
  stop("Fila inválida. Ingresa un número entre 1 y ", nrow(log_modelos), ".")
}
modelo_sel <- log_modelos[fila_sel, ]

cat("\n  Modelo seleccionado:\n")
print(modelo_sel)

ruta_rds <- here(paths$submissions, modelo_sel$tipo, paste0(modelo_sel$nombre, ".rds"))
if (!file.exists(ruta_rds)) {
  stop("No se encontró el .rds del modelo en: ", ruta_rds)
}
modelo <- readRDS(ruta_rds)

# ------------------------------------------------------------
# 2. CARGA DE DATOS
# ------------------------------------------------------------

train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

# ------------------------------------------------------------
# 3. EXTRAER PROBABILIDADES Y OBSERVADO
# ------------------------------------------------------------

extract_pred <- function(modelo, train, ruta_preds = NULL) {
  # XGBoost (xgb.Booster): lee OOF preds guardadas en _train_preds.rds
  if (inherits(modelo, "xgb.Booster")) {
    if (is.null(ruta_preds) || !file.exists(ruta_preds)) {
      stop("XGBoost requiere el archivo _train_preds.rds. ",
           "Ruta esperada: ", ruta_preds)
    }
    probs <- as.numeric(readRDS(ruta_preds))
    return(list(
      probs  = probs,
      obs    = as.integer(train$pobre == 1),
      fold   = NA_character_,
      source = "OOF (xgb.cv)"
    ))
  }
  # Random Forest (ranger directo): OOB
  if (inherits(modelo, "ranger")) {
    return(list(
      probs  = as.numeric(modelo$predictions[, "pobre"]),
      obs    = as.integer(train$pobre == 1),
      fold   = NA_character_,
      source = "OOB (ranger)"
    ))
  }
  # LPM (lm): in-sample (no hay clase para predecir prob)
  if (!is.null(modelo$method) && modelo$method == "lm") {
    yhat <- pmin(pmax(predict(modelo, train), 0), 1)
    return(list(
      probs  = as.numeric(yhat),
      obs    = as.integer(train$pobre == 1),
      fold   = NA_character_,
      source = "In-sample (LPM)"
    ))
  }
  # caret default: usa $pred filtrado por bestTune
  bt   <- modelo$bestTune
  pred <- modelo$pred
  if (is.null(pred)) {
    stop("modelo$pred está vacío — entrena con savePredictions = 'final'")
  }
  if (!is.null(bt) && nrow(bt) > 0) {
    for (col in names(bt)) {
      pred <- pred[pred[[col]] == bt[[col]], , drop = FALSE]
    }
  }
  list(
    probs  = as.numeric(pred$pobre),
    obs    = as.integer(pred$obs == "pobre"),
    fold   = if ("Resample" %in% names(pred)) pred$Resample else NA_character_,
    source = sprintf("CV (%s)", modelo$method)
  )
}

ruta_preds <- here(paths$submissions, modelo_sel$tipo,
                   paste0(modelo_sel$nombre, "_train_preds.rds"))
P         <- extract_pred(modelo, train, ruta_preds)
threshold <- as.numeric(modelo_sel$threshold)

cat(sprintf("\n  Source predicciones: %s\n", P$source))
cat(sprintf("  Threshold:           %.4f\n", threshold))

# ------------------------------------------------------------
# 4. HELPERS DE MÉTRICAS
# ------------------------------------------------------------

f1_at <- function(p, y, t) {
  pr   <- as.integer(p >= t)
  tp   <- sum(pr == 1 & y == 1)
  fp   <- sum(pr == 1 & y == 0)
  fn   <- sum(pr == 0 & y == 1)
  tn   <- sum(pr == 0 & y == 0)
  prec <- if (tp + fp == 0) 0 else tp / (tp + fp)
  rec  <- if (tp + fn == 0) 0 else tp / (tp + fn)
  f1   <- if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
  list(precision = prec, recall = rec, f1 = f1,
       tp = tp, fp = fp, fn = fn, tn = tn)
}

# Yardstick necesita factor con event_level segundo
df_yard <- tibble(probs = P$probs,
                  obs   = factor(P$obs, levels = c(0, 1)))

auc_roc <- yardstick::roc_auc(df_yard, obs, probs, event_level = "second")$.estimate
auc_pr  <- yardstick::pr_auc (df_yard, obs, probs, event_level = "second")$.estimate
df_pr   <- yardstick::pr_curve (df_yard, obs, probs, event_level = "second")
df_roc  <- yardstick::roc_curve(df_yard, obs, probs, event_level = "second")

# ------------------------------------------------------------
# 5. TABLAS
# ------------------------------------------------------------

m_at <- f1_at(P$probs, P$obs, threshold)

# 5.1 Métricas globales
tabla_metricas <- tibble(
  metrica = c("AUC-ROC", "AUC-PR", "Precision", "Recall", "F1",
              "Accuracy", "Threshold", "Tasa base (Pobre)"),
  valor   = c(auc_roc, auc_pr, m_at$precision, m_at$recall, m_at$f1,
              (m_at$tp + m_at$tn) / length(P$probs),
              threshold, mean(P$obs))
) |> mutate(valor = round(valor, 4))

write.csv(tabla_metricas,
          here(paths$tables, "deck1_metricas_globales.csv"),
          row.names = FALSE)

# 5.2 Confusion matrix
cm <- tibble(
  obs  = c("Pobre",    "Pobre",    "No pobre", "No pobre"),
  pred = c("Pobre",    "No pobre", "Pobre",    "No pobre"),
  n    = c(m_at$tp,    m_at$fn,    m_at$fp,    m_at$tn)
) |> mutate(pct = round(n / sum(n) * 100, 2))

write.csv(cm, here(paths$tables, "deck1_confusion_matrix.csv"),
          row.names = FALSE)

# 5.3 Sweep de threshold
ts    <- seq(0.05, 0.95, by = 0.005)
sweep <- map_dfr(ts, function(t) {
  m <- f1_at(P$probs, P$obs, t)
  tibble(threshold = t, precision = m$precision,
         recall = m$recall, f1 = m$f1)
})
write.csv(sweep, here(paths$tables, "deck1_sweep_threshold.csv"),
          row.names = FALSE)

# 5.4 F1 por fold (si hubo CV)
f1_folds <- NULL
if (!any(is.na(P$fold))) {
  f1_folds <- tibble(probs = P$probs, obs = P$obs, fold = P$fold) |>
    group_by(fold) |>
    group_modify(~ {
      m <- f1_at(.x$probs, .x$obs, threshold)
      tibble(precision = m$precision, recall = m$recall, f1 = m$f1, n = nrow(.x))
    }) |> ungroup()
  write.csv(f1_folds, here(paths$tables, "deck1_f1_por_fold.csv"),
            row.names = FALSE)
}

# 5.5 Calibration (deciles)
calib <- tibble(probs = P$probs, obs = P$obs) |>
  mutate(decil = ntile(probs, 10)) |>
  group_by(decil) |>
  summarise(prob_media = mean(probs),
            tasa_obs   = mean(obs),
            n          = n(),
            .groups    = "drop")
write.csv(calib, here(paths$tables, "deck1_calibration.csv"),
          row.names = FALSE)

# 5.6 Variable importance (si aplica)
varimp <- NULL
if (inherits(modelo, "ranger") && !is.null(modelo$variable.importance) &&
    length(modelo$variable.importance) > 0) {
  varimp <- tibble(variable   = names(modelo$variable.importance),
                   importance = as.numeric(modelo$variable.importance))
} else if (!is.null(modelo$method) && modelo$method != "lm") {
  vi <- tryCatch(caret::varImp(modelo)$importance, error = function(e) NULL)
  if (!is.null(vi)) {
    varimp <- tibble(variable = rownames(vi),
                     importance = vi[[1]])
  }
}
if (!is.null(varimp)) {
  varimp <- varimp |> arrange(desc(importance))
  write.csv(varimp, here(paths$tables, "deck1_var_importance.csv"),
            row.names = FALSE)
}

# 5.7 Comparación CV vs Kaggle (gap de overfitting)
gap_tbl <- tibble(
  metrica   = c("CV F1", "Kaggle F1", "Gap (CV - Kaggle)"),
  valor     = c(modelo_sel$cv_f1,
                modelo_sel$kaggle_f1,
                if (is.na(modelo_sel$kaggle_f1)) NA_real_
                else modelo_sel$cv_f1 - modelo_sel$kaggle_f1)
) |> mutate(valor = round(valor, 4))
write.csv(gap_tbl, here(paths$tables, "deck1_cv_vs_kaggle.csv"),
          row.names = FALSE)

# ------------------------------------------------------------
# 6. FIGURAS
# ------------------------------------------------------------

tema <- theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", size = 14, color = "#0A2240"),
        plot.subtitle = element_text(size = 11, color = "#555"),
        plot.caption  = element_text(size = 9, color = "#888", hjust = 0),
        legend.position = "bottom",
        panel.grid.minor = element_blank())

guardar_fig <- function(p, nombre, w = 9, h = 6) {
  ruta <- here(paths$figures, paste0(nombre, ".png"))
  ggsave(ruta, p, width = w, height = h, dpi = 300, bg = "white")
  cat(sprintf("    [fig] %s\n", basename(ruta)))
}

caption_modelo <- sprintf("Modelo: %s / %s | %s",
                          modelo_sel$tipo, modelo_sel$nombre, P$source)

# 6.1 PR curve
p_pr <- df_pr |>
  ggplot(aes(recall, precision)) +
  geom_path(color = "#B71C1C", linewidth = 1.1) +
  geom_hline(yintercept = mean(P$obs), linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.65, y = mean(P$obs) + 0.025,
           label = sprintf("Baseline = %.2f", mean(P$obs)),
           color = "gray50", size = 3.5) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title    = "Curva Precision-Recall",
       subtitle = sprintf("AUC-PR = %.3f (relevante con desbalance ~%.0f/%.0f)",
                          auc_pr, (1 - mean(P$obs)) * 100, mean(P$obs) * 100),
       x = "Recall", y = "Precision",
       caption = caption_modelo) +
  tema
guardar_fig(p_pr, "deck1_01_pr_curve")

# 6.2 ROC curve (complementaria)
p_roc <- df_roc |>
  ggplot(aes(1 - specificity, sensitivity)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_path(color = "#1565C0", linewidth = 1.1) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title    = "Curva ROC",
       subtitle = sprintf("AUC-ROC = %.3f", auc_roc),
       x = "1 - Specificity", y = "Sensitivity (Recall)",
       caption = caption_modelo) +
  tema
guardar_fig(p_roc, "deck1_02_roc_curve")

# 6.3 F1 vs threshold
p_sweep <- sweep |>
  pivot_longer(c(precision, recall, f1), names_to = "metrica") |>
  mutate(metrica = factor(metrica, levels = c("precision", "recall", "f1"),
                          labels = c("Precision", "Recall", "F1"))) |>
  ggplot(aes(threshold, value, color = metrica)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = threshold, linetype = "dashed", color = "black") +
  geom_vline(xintercept = 0.5,       linetype = "dotted", color = "gray60") +
  annotate("text", x = threshold + 0.015, y = 0.05,
           label = sprintf("t* = %.3f", threshold), hjust = 0, size = 3.5) +
  annotate("text", x = 0.515, y = 0.05,
           label = "0.5", hjust = 0, color = "gray60", size = 3.5) +
  scale_color_manual(values = c(Precision = "#1565C0",
                                Recall    = "#E65100",
                                F1        = "#B71C1C")) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title    = "Precision, Recall y F1 vs threshold",
       subtitle = sprintf("F1 máximo en t* = %.3f (vs default 0.5)", threshold),
       x = "Threshold", y = "Score", color = NULL,
       caption = caption_modelo) +
  tema
guardar_fig(p_sweep, "deck1_03_f1_vs_threshold")

# 6.4 Distribución de probabilidades por clase
p_dist <- tibble(probs = P$probs,
                 clase = factor(P$obs, levels = c(0, 1),
                                labels = c("No pobre", "Pobre"))) |>
  ggplot(aes(probs, fill = clase)) +
  geom_density(alpha = 0.5, color = NA) +
  geom_vline(xintercept = threshold, linetype = "dashed") +
  scale_fill_manual(values = c("No pobre" = "#1565C0",
                               "Pobre"    = "#B71C1C")) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(title    = "Distribución de probabilidades predichas por clase real",
       subtitle = "Mayor separación entre densidades = mejor capacidad discriminativa",
       x = "P(pobre)", y = "Densidad", fill = NULL,
       caption = caption_modelo) +
  tema
guardar_fig(p_dist, "deck1_04_distribucion_probs")

# 6.5 Calibration plot
p_cal <- calib |>
  ggplot(aes(prob_media, tasa_obs)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(color = "#B71C1C", linewidth = 1) +
  geom_point(size = 3, color = "#B71C1C") +
  geom_text(aes(label = decil), vjust = -1.2, size = 3, color = "#555") +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title    = "Calibración (deciles de probabilidad)",
       subtitle = "La línea diagonal es la calibración perfecta",
       x = "Probabilidad predicha (media decil)",
       y = "Tasa observada en el decil",
       caption = caption_modelo) +
  tema
guardar_fig(p_cal, "deck1_05_calibration")

# 6.6 Variable importance top 15
if (!is.null(varimp) && nrow(varimp) > 0) {
  vi_top <- head(varimp, 15) |>
    mutate(variable = fct_reorder(variable, importance))
  p_vi <- vi_top |>
    ggplot(aes(importance, variable)) +
    geom_col(fill = "#1565C0", alpha = 0.85) +
    geom_text(aes(label = round(importance, 3)), hjust = -0.1, size = 3.2) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title    = "Top 15 variables por importancia",
         subtitle = "Mayor importancia = mayor contribución a la predicción",
         x = "Importance", y = NULL,
         caption = caption_modelo) +
    tema
  guardar_fig(p_vi, "deck1_06_var_importance", h = 7)
}

# 6.7 Confusion matrix heatmap
cm_plot <- cm |>
  mutate(obs  = factor(obs,  levels = c("No pobre", "Pobre")),
         pred = factor(pred, levels = c("No pobre", "Pobre")))
p_cm <- cm_plot |>
  ggplot(aes(pred, obs, fill = n)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%s\n(%.1f%%)",
                                scales::comma(n), pct)),
            size = 6, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "#90A4AE", high = "#0D47A1") +
  labs(title    = "Matriz de confusión",
       subtitle = sprintf("threshold = %.3f | n = %s",
                          threshold, scales::comma(sum(cm$n))),
       x = "Predicción", y = "Observado", fill = "n",
       caption = caption_modelo) +
  tema
guardar_fig(p_cm, "deck1_07_confusion_matrix", w = 7, h = 6)

# 6.8 F1 por fold (si hubo CV)
if (!is.null(f1_folds)) {
  p_fold <- f1_folds |>
    mutate(fold = reorder(fold, f1)) |>
    ggplot(aes(fold, f1)) +
    geom_col(fill = "#B71C1C", alpha = 0.85) +
    geom_hline(yintercept = mean(f1_folds$f1),
               linetype = "dashed", color = "gray30") +
    geom_text(aes(label = sprintf("%.3f", f1)), vjust = -0.4, size = 3.5) +
    scale_y_continuous(limits = c(0, max(f1_folds$f1) * 1.15)) +
    labs(title    = sprintf("F1 por fold (media = %.3f, sd = %.3f)",
                            mean(f1_folds$f1), sd(f1_folds$f1)),
         subtitle = "Variabilidad del modelo entre folds del CV",
         x = "Fold", y = "F1",
         caption = caption_modelo) +
    tema
  guardar_fig(p_fold, "deck1_08_f1_por_fold", w = 8, h = 5)
}

# 6.9 CV vs Kaggle (gap)
if (!is.na(modelo_sel$kaggle_f1)) {
  gap_long <- tibble(
    set = c("CV (interno)", "Kaggle (público)"),
    f1  = c(modelo_sel$cv_f1, modelo_sel$kaggle_f1)
  )
  p_gap <- gap_long |>
    ggplot(aes(set, f1, fill = set)) +
    geom_col(width = 0.5, show.legend = FALSE, alpha = 0.9) +
    geom_text(aes(label = sprintf("%.4f", f1)),
              vjust = -0.3, size = 4.5, fontface = "bold") +
    scale_fill_manual(values = c("CV (interno)"     = "#1565C0",
                                 "Kaggle (público)" = "#B71C1C")) +
    scale_y_continuous(limits = c(0, max(gap_long$f1) * 1.2)) +
    labs(title    = sprintf("CV F1 vs Kaggle F1 (gap = %.4f)",
                            modelo_sel$cv_f1 - modelo_sel$kaggle_f1),
         subtitle = "Gap positivo grande = posible overfitting al training",
         x = NULL, y = "F1",
         caption = caption_modelo) +
    tema
  guardar_fig(p_gap, "deck1_09_cv_vs_kaggle", w = 6, h = 5)
}

# ------------------------------------------------------------
# 7. RESUMEN EN CONSOLA
# ------------------------------------------------------------

cat("\n================ RESUMEN ================\n")
cat(sprintf("  Tipo / Nombre:  %s / %s\n", modelo_sel$tipo, modelo_sel$nombre))
cat(sprintf("  Threshold:      %.4f\n", threshold))
cat(sprintf("  CV F1:          %.4f\n", modelo_sel$cv_f1))
cat(sprintf("  Kaggle F1:      %s\n",
            ifelse(is.na(modelo_sel$kaggle_f1), "—",
                   sprintf("%.4f", modelo_sel$kaggle_f1))))
cat(sprintf("  AUC-ROC:        %.4f\n", auc_roc))
cat(sprintf("  AUC-PR:         %.4f\n", auc_pr))
cat(sprintf("  Source probs:   %s\n",   P$source))
cat("\n  Tablas:  ", here(paths$tables),  "\n")
cat("  Figuras: ", here(paths$figures), "\n")
cat("\n  Activos generados (prefijo deck1_):\n")
cat("    Tablas:\n")
cat("      - deck1_metricas_globales.csv\n")
cat("      - deck1_confusion_matrix.csv\n")
cat("      - deck1_sweep_threshold.csv\n")
cat("      - deck1_calibration.csv\n")
cat("      - deck1_cv_vs_kaggle.csv\n")
if (!is.null(f1_folds))   cat("      - deck1_f1_por_fold.csv\n")
if (!is.null(varimp))     cat("      - deck1_var_importance.csv\n")
cat("    Figuras:\n")
cat("      - deck1_01_pr_curve.png\n")
cat("      - deck1_02_roc_curve.png\n")
cat("      - deck1_03_f1_vs_threshold.png\n")
cat("      - deck1_04_distribucion_probs.png\n")
cat("      - deck1_05_calibration.png\n")
if (!is.null(varimp))     cat("      - deck1_06_var_importance.png\n")
cat("      - deck1_07_confusion_matrix.png\n")
if (!is.null(f1_folds))   cat("      - deck1_08_f1_por_fold.png\n")
if (!is.na(modelo_sel$kaggle_f1)) cat("      - deck1_09_cv_vs_kaggle.png\n")

# ------------------------------------------------------------
# 8. LIMPIAR
# ------------------------------------------------------------
rm(extract_pred, f1_at, guardar_fig, ts, df_yard,
   df_pr, df_roc, sweep, calib, cm, cm_plot, m_at,
   tabla_metricas, gap_tbl, log_modelos)
gc()
