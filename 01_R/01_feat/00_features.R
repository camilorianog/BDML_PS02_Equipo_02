# ============================================================
# 00_features.R
# Feature engineering con respaldo literario
# ============================================================

train <- readRDS(here(paths$processed, "train_clean.rds"))
test  <- readRDS(here(paths$processed, "test_clean.rds"))

feature_engineer <- function(df) {
  df |>
    mutate(
      
      # --- 1. Ratio de dependencia -------------------------
      # (IDB Costa Rica Kaggle 2018)
      ratio_dependencia = (n_menores_18 + n_mayores_65) /
        pmax(nper - n_menores_18 - n_mayores_65, 1),
      
      # --- 2. Hacinamiento --------------------------------
      # (Banerjee 2018, Mentalbreaks 2019)
      hacinamiento      = nper / pmax(p5000, 1),
      
      # --- 3. Cuartos por persona -------------------------
      # (Browne et al. 2018)
      cuartos_per_cap   = p5000 / pmax(nper, 1),
      
      # --- 4. Interacción educación × ocupación -----------
      # (Nkurunziza et al. 2024, Marrugo-Arnedo et al. 2015)
      educ_x_ocup       = as.integer(nivel_educ_max) * tasa_ocupacion,
      
      # --- 5. Interacción zona × ocupación ----------------
      # (Obando Rozo & Andrián 2015, World Bank 2019)
      rural_x_ocup      = as.integer(clase == "2") * tasa_ocupacion,
      
      # --- 6. Interacción formalidad × seguridad social ---
      # (World Bank 2019, UNDP & ECLAC 2024)
      formal_x_salud    = prop_cotiza_pension * prop_afiliado_salud,
      
      # --- 7. Polinomio tamaño del hogar ------------------
      # (UN Statistics 2005, SOAS 2005)
      nper_sq     = nper^2,
      
      # --- 8. Polinomio edad promedio ---------------------
      # (Obando Rozo & Andrián 2015, Banerjee 2018)
      edad_prom_sq      = edad_promedio^2,
      
      # --- 9. Interacción género × inactividad ------------
      # (Corral et al. 2024, UNDP & ECLAC 2024)
      mujeres_x_inact   = prop_mujeres * (1 - tasa_ocupacion),
      
      # --- 10. Jefatura femenina × inactividad ------------
      # (Bleynat et al. 2020, Chant 2003, GEIH 2018)
      jefe_mujer_inact  = jefe_mujer * (1 - tasa_ocupacion)
    )
}

cat(">>> Aplicando feature engineering...\n")
train <- feature_engineer(train)
test  <- feature_engineer(test)

# --- Guardar --------------------------------------------
saveRDS(train, here(paths$processed, "train_features.rds"))
saveRDS(test,  here(paths$processed, "test_features.rds"))

# --- Limpiar entorno ------------------------------------
rm(feature_engineer)
gc()

cat(">>> 00_features.R completado\n")
cat("    train:", nrow(train), "filas x", ncol(train), "columnas\n")
cat("    test: ", nrow(test),  "filas x", ncol(test),  "columnas\n")