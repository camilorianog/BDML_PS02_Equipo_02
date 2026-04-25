# ============================================================
# 00_features.R
# Feature engineering para predicción de pobreza en hogares
#
# Fuentes generales del marco conceptual:
#   - IDB / Kaggle Costa Rica Poverty (2018): benchmark regional
#   - Banerjee & Duflo "Poor Economics" (2011/2018): patrones
#     de consumo y trabajo en hogares pobres
#   - World Bank "Poverty and Shared Prosperity" (2019):
#     dimensiones multidimensionales de vulnerabilidad
#   - UNDP & ECLAC "Social Panorama of Latin America" (2024):
#     brechas de género, informalidad y protección social
#   - Corral et al. "Inequality in a Lower Middle Income
#     Country" (2024): Colombia específicamente
#   - Obando Rozo & Andrián (2015): mercado laboral urbano-
#     rural en Colombia
#   - Nkurunziza et al. (2024): capital humano y pobreza
#   - Marrugo-Arnedo et al. (2015): educación y pobreza rural
#   - Bleynat et al. (2020) & Chant (2003): feminización
#     de la pobreza y jefatura femenina
#   - UN Statistics (2005) & SOAS (2005): tamaño del hogar
#     y ciclo de vida
# ============================================================

train <- readRDS(here(paths$processed, "train_clean.rds"))
test  <- readRDS(here(paths$processed, "test_clean.rds"))

feature_engineer <- function(df) {
  df |>
    mutate(
      
      # --- 1. Ratio de dependencia -------------------------
      # Fracción del hogar que no genera ingreso laboral.
      # A mayor ratio, mayor presión sobre los ocupados.
      # (IDB Costa Rica Kaggle 2018)
      ratio_dependencia = (n_personas - n_ocupados) /
        pmax(n_personas, 1),
      
      # --- 2. Hacinamiento --------------------------------
      # Personas por cuarto total: proxy de calidad habitacional
      # y densidad del hogar. Umbral crítico ≥ 3 pers/cuarto.
      # (Banerjee 2018, Browne et al. 2018, Mentalbreaks 2019)
      hacinamiento = n_personas / pmax(cuartos_hogar, 1),
      
      # --- 3. Interacción educación × ocupación -----------
      # Capital humano solo genera valor si hay empleo.
      # Un hogar educado pero desempleado no escapa la pobreza.
      # (Nkurunziza et al. 2024, Marrugo-Arnedo et al. 2015)
      educ_x_ocup = as.integer(nivel_educ_max) * tasa_ocupacion,
      
      # --- 4. Interacción zona × ocupación ----------------
      # En zonas rurales el empleo tiene menor productividad
      # y mayor informalidad; la interacción captura esa brecha.
      # (Obando Rozo & Andrián 2015, World Bank 2019)
      # NOTA: usa cabecera (renombrada de clase en 00_clean.R)
      rural_x_ocup = as.integer(cabecera == "2") * tasa_ocupacion,
      
      # --- 5. Interacción formalidad × seguridad social ---
      # Formalidad plena = cotiza pensión Y tiene salud contributiva.
      # Proxy de inserción real en el mercado formal.
      # (World Bank 2019, UNDP & ECLAC 2024)
      formal_x_salud = prop_cotiza_pension * prop_afiliado_salud,
      
      # --- 6. Polinomio tamaño del hogar ------------------
      # Rendimientos decrecientes: pasar de 3 a 4 personas
      # no impacta igual que pasar de 7 a 8.
      # (UN Statistics 2005, SOAS 2005)
      n_personas_sq = n_personas^2,
      
      # --- 7. Polinomio edad promedio ---------------------
      # Relación en U invertida entre edad y pobreza:
      # hogares muy jóvenes o muy viejos son más vulnerables.
      # (Obando Rozo & Andrián 2015, Banerjee 2018)
      edad_prom_sq = edad_promedio^2,
      
      # --- 8. Interacción género × inactividad ------------
      # Hogares con más mujeres y baja ocupación combinan
      # la brecha de participación laboral femenina con
      # la falta de ingreso.
      # (Corral et al. 2024, UNDP & ECLAC 2024)
      mujeres_x_inact = prop_mujeres * (1 - tasa_ocupacion),
      
      # --- 9. Jefatura femenina × inactividad ------------
      # La jefatura femenina per se no implica pobreza, pero
      # sí cuando se combina con baja inserción laboral del hogar.
      # (Bleynat et al. 2020, Chant 2003, GEIH 2018)
      jefe_mujer_inact = jefe_mujer * (1 - tasa_ocupacion),
      
      # --- 10. Tasa de inactividad ------------------------
      # Fracción de la PET que no busca ni tiene empleo.
      # Complementa ratio_dependencia con foco en la PET.
      tasa_inactivos = n_inactivos / pmax(n_pet, 1),
      
      # --- 11. Dummy sin ocupados -------------------------
      # Hogar con cero ocupados: umbral duro de vulnerabilidad.
      # Captura no linealidades de ratio_dependencia en torno a 1.
      sin_ocupados = as.integer(n_ocupados == 0),
      
      # --- 12. Educación jefe × ocupación jefe ------------
      # El jefe es el principal generador de ingreso en la mayoría
      # de los hogares; su capital humano solo "activa" si trabaja.
      educ_jefe_x_ocup = as.integer(educ_jefe) * ocup_jefe,
      
      # --- 13. Calidad del empleo -------------------------
      # Horas trabajadas × formalidad: proxy de ingreso laboral
      # implícito sin usar los montos directamente (leakage).
      # Un ocupado informal de pocas horas ≠ uno formal full-time.
      # (World Bank 2019, UNDP & ECLAC 2024)
      calidad_empleo = horas_trabajo_prom * prop_formal,
      
      # --- 14. Presión habitacional -----------------------
      # Cuartos efectivamente usados sobre cuartos disponibles.
      # Complementa hacinamiento: un hogar puede tener cuartos
      # pero no usarlos (subarriendo, abandono).
      # (UN-Habitat 2020)
      presion_habitacional = cuartos_ocupados / pmax(cuartos_hogar, 1),
      
      # --- 15. Jefe vulnerable ----------------------------
      # Jefe sin empleo y sin educación más allá de primaria:
      # combina dos de los predictores más fuertes de pobreza crónica.
      # (Corral et al. 2024, Nkurunziza et al. 2024)
      jefe_vulnerable = as.integer(ocup_jefe == 0 & educ_jefe <= 3),
      
      # --- 16. Doble protección social --------------------
      # Alta formalidad + alta cotización pensional:
      # hogar con cobertura laboral completa, señal fuerte
      # de ingreso estable por encima de la línea de pobreza.
      # (UNDP & ECLAC 2024)
      doble_proteccion = prop_formal * prop_cotiza_pension,
      
      # --- 17. Ratio adultos mayores sobre PET ------------
      # Carga específica de vejez dentro de la población activa.
      # Distinto de n_mayores_65: normaliza por el tamaño de la PET.
      # (World Bank 2019)
      ratio_mayores_65 = n_mayores_65 / pmax(n_pet, 1),
      
      # --- 18. Jefe mayor inactivo ------------------------
      # Jefe mayor de 60 años sin empleo: alta probabilidad de
      # depender de transferencias o de otros miembros del hogar.
      # (Banerjee 2018, GEIH metodología 2023)
      jefe_mayor_inactivo = as.integer(edad_jefe > 60 & ocup_jefe == 0),
      
      # --- 19. Subempleo × informalidad -------------------
      # Doble penalización: quieren trabajar más Y lo hacen
      # en condiciones informales. Máxima vulnerabilidad laboral.
      # (OIT "World Employment and Social Outlook" 2023)
      subempleo_informal = tasa_subempleo * (1 - prop_formal),
      
      # --- 20. Estabilidad formal -------------------------
      # Antigüedad del jefe × dummy formal:
      # si el jefe lleva tiempo en un empleo formal, la
      # estabilidad del ingreso del hogar es mucho mayor.
      # (World Bank 2019, Corral et al. 2024)
      estabilidad_formal = antiguedad_jefe * jefe_formal
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
cat("\n    Features nuevas [v2]:\n")
cat("    [13] calidad_empleo\n")
cat("    [14] presion_habitacional\n")
cat("    [15] jefe_vulnerable\n")
cat("    [16] doble_proteccion\n")
cat("    [17] ratio_mayores_65\n")
cat("    [18] jefe_mayor_inactivo\n")
cat("    [19] subempleo_informal\n")
cat("    [20] estabilidad_formal\n")
cat("    [Fix] nper → n_personas, p5000 → cuartos_hogar (nombres post-clean)\n")