# ============================================================
# 00_optimizar_threshold.R
# Optimiza el threshold de clasificación para maximizar F1
# ============================================================

optimizar_threshold <- function(modelo, datos, target) {
  probs      <- predict(modelo, datos, type = "prob")[, "pobre"]
  thresholds <- seq(0.1, 0.9, by = 0.01)
  
  f1_scores <- map_dbl(thresholds, function(t) {
    preds <- factor(ifelse(probs >= t, "pobre", "no_pobre"),
                    levels = c("no_pobre", "pobre"))
    cm <- confusionMatrix(preds, target, positive = "pobre")
    cm$byClass["F1"]
  })
  
  best_t  <- thresholds[which.max(f1_scores)]
  best_f1 <- max(f1_scores, na.rm = TRUE)
  
  list(threshold = best_t, f1 = best_f1)
}