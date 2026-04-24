# ============================================================
# 00_features_v2.R
# Feature engineering sobre la base REDUCIDA (clean_v2)
# ------------------------------------------------------------
# Solo features de alto impacto confirmado por literatura.
# Inputs esperados: train_clean.rds / test_clean.rds producidos
# por 00_clean_v2.R (núcleo simple).
# ============================================================

train <- readRDS(here(paths$processed, "train_clean.rds"))
test  <- readRDS(here(paths$processed, "test_clean.rds"))

feature_engineer_v2 <- function(df) {
  df |>
    mutate(

      # --- 1. Hacinamiento --------------------------------
      # Personas por cuarto; uno de los predictores más
      # consistentes de pobreza en la literatura.
      # (Banerjee 2018; Browne et al. 2018; Mentalbreaks 2019)
      hacinamiento = n_personas / pmax(cuartos_hogar, 1),

      # --- 2. Ratio de dependencia -----------------------
      # Fracción del hogar que NO aporta ingreso laboral.
      # (IDB Costa Rica Kaggle 2018)
      ratio_dependencia = (n_personas - n_ocupados) /
                           pmax(n_personas, 1),

      # --- 3. Dummy sin ocupados -------------------------
      # Captura la no-linealidad del ratio de dependencia
      # cuando se acerca a 1. Hogares sin ningún ocupado
      # tienen tasa de pobreza mucho mayor.
      sin_ocupados = as.integer(n_ocupados == 0),

      # --- 4. Educación × ocupación (hogar) --------------
      # Capital humano activo — interacción clave en
      # modelos de pobreza de Colombia.
      # (Nkurunziza et al. 2024; Marrugo-Arnedo et al. 2015)
      educ_x_ocup = as.integer(nivel_educ_max) * tasa_ocupacion,

      # --- 5. Educación × ocupación (jefe) ---------------
      # Análogo de (4) pero aplicado al jefe del hogar.
      educ_jefe_x_ocup = as.integer(educ_jefe) * ocup_jefe,

      # --- 6. Polinomio tamaño del hogar -----------------
      # Relación no-lineal entre tamaño y pobreza: hogares
      # muy grandes y muy pequeños tienen dinámicas distintas.
      # (UN Statistics 2005; SOAS 2005)
      nper_sq = n_personas^2

    )
}

cat(">>> Aplicando feature engineering v2 (alto impacto)...\n")
train <- feature_engineer_v2(train)
test  <- feature_engineer_v2(test)

# --- Guardar (sufijo v2 para preservar original) ------------
saveRDS(train, here(paths$processed, "train_features_v2.rds"))
saveRDS(test,  here(paths$processed, "test_features_v2.rds"))

cat(">>> 00_features_v2.R completado\n")
cat("    train:", nrow(train), "x", ncol(train), "\n")
cat("    test: ", nrow(test),  "x", ncol(test),  "\n")
cat("    Features agregados: hacinamiento, ratio_dependencia,\n")
cat("                        sin_ocupados, educ_x_ocup,\n")
cat("                        educ_jefe_x_ocup, nper_sq\n")

# --- Limpiar entorno ----------------------------------------
rm(feature_engineer_v2)
gc()
