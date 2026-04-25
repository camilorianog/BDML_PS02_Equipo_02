# ============================================================
# 00_optimizar_threshold.R
# ============================================================

optimizar_threshold <- function(modelo, dados, target) {
  
  if (inherits(modelo, "ranger")) {
    probs      <- modelo$predictions[, "pobre"]
    target_bin <- as.integer(target == "pobre")
    
  } else if (inherits(modelo, "xgb.Booster")) {
    X          <- predict(dummy_recipe,
                          dados |> select(-id, -any_of("pobre")))
    probs      <- predict(modelo, xgb.DMatrix(data = X))
    target_bin <- as.integer(target == "pobre")
    
  } else if (modelo$method == "lm") {
    probs      <- pmin(pmax(predict(modelo, dados), 0), 1)
    target_bin <- as.integer(target == 1)
    
  } else {
    probs      <- predict(modelo, dados, type = "prob")[, "pobre"]
    target_bin <- as.integer(target == "pobre")
  }
  
  thresholds <- seq(0.25, 0.55, by = 0.005)
  
  f1_scores <- map_dbl(thresholds, function(t) {
    preds     <- as.integer(probs >= t)
    tp        <- sum(preds == 1 & target_bin == 1)
    fp        <- sum(preds == 1 & target_bin == 0)
    fn        <- sum(preds == 0 & target_bin == 1)
    precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
    recall    <- if (tp + fn == 0) 0 else tp / (tp + fn)
    if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
  })
  
  best_t  <- thresholds[which.max(f1_scores)]
  best_f1 <- max(f1_scores, na.rm = TRUE)
  
  list(threshold = best_t, f1 = best_f1)
}
  
  best_t  <- thresholds[which.max(f1_scores)]
  best_f1 <- max(f1_scores, na.rm = TRUE)
  
  list(threshold = best_t, f1 = best_f1)
}
