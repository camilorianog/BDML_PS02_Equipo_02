# ============================================================
# 00_clean.R
# Limpieza general del dataset
# ============================================================

# --- Verificar datos ----------------------------------------
archivos_necesarios <- c(
  "train_hogares.csv", "test_hogares.csv",
  "train_personas.csv", "test_personas.csv"
)

faltantes <- archivos_necesarios[
  !file.exists(here(paths$raw, archivos_necesarios))
]

if (length(faltantes) > 0) {
  stop(
    "\n========================================================\n",
    "  Faltan archivos en 00_data/00_raw/:\n",
    paste(" -", faltantes, collapse = "\n"), "\n\n",
    "  Descárgalos desde la terminal con:\n\n",
    "  kaggle competitions download -c uniandes-bdml-2026-10-ps-2 \\\n",
    "    -p ", here(paths$raw), "\n\n",
    "  Luego descomprime el zip en la misma carpeta.\n",
    "========================================================\n"
  )
} else {
  cat(">>> Datos encontrados")
}

# --- Cargar datos -------------------------------------------
train_h <- read.csv(here(paths$raw, "train_hogares.csv"))
test_h  <- read.csv(here(paths$raw, "test_hogares.csv"))
train_p <- read.csv(here(paths$raw, "train_personas.csv"))
test_p  <- read.csv(here(paths$raw, "test_personas.csv"))

cat("Dims originales:\n")
cat("  train_hogares:", dim(train_h), "\n")
cat("  test_hogares: ", dim(test_h),  "\n")
cat("  train_personas:", dim(train_p), "\n")
cat("  test_personas: ", dim(test_p),  "\n")

# --- Variables a excluir ------------------------------------

excluir_hogares <- c(
  "Ingtotug", "Ingtotugarr", "Ingpcug",
  "Lp", "Li",
  "Indigente", "Npobres", "Nindigentes",
  "Fex_c", "Fex_dpto", "Mes",
  # Montos de vivienda — capturados por p5090
  "P5100", "P5130", "P5140"
)

excluir_personas <- c(
  # Estrato — removido del test
  "Estrato1",
  # Salarios y componentes de ingreso laboral
  "P6500", "P6510s1", "P6510s2",
  "P6545s1", "P6545s2",
  "P6580s1", "P6580s2",
  "P6585s1a1", "P6585s1a2",
  "P6585s2a1", "P6585s2a2",
  "P6585s3a1", "P6585s3a2",
  "P6585s4a1", "P6585s4a2",
  "P6590s1", "P6600s1", "P6610s1", "P6620s1",
  "P6630s1a1", "P6630s2a1", "P6630s3a1",
  "P6630s4a1", "P6630s6a1",
  "P6750", "P6760", "P550", "P7070",
  "P7140s1", "P7140s2",
  "P7422s1", "P7472s1",
  "P7500s1", "P7500s1a1", "P7500s2a1", "P7500s3a1",
  "P7510s1a1", "P7510s2a1", "P7510s3a1",
  "P7510s5a1", "P7510s6a1", "P7510s7a1",
  # Ingresos agregados e imputados
  "Impa", "Isa", "Ie", "Imdi",
  "Iof1", "Iof2", "Iof3h", "Iof3i", "Iof6",
  # Flags de imputación
  "Cclasnr2", "Cclasnr3", "Cclasnr4", "Cclasnr5",
  "Cclasnr6", "Cclasnr7", "Cclasnr8", "Cclasnr11",
  # Versiones imputadas
  "Impaes", "Isaes", "Iees", "Imdies",
  "Iof1es", "Iof2es", "Iof3hes", "Iof3ies", "Iof6es",
  # Ingresos totales
  "Ingtotob", "Ingtotes", "Ingtot",
  # Administrativas
  "Fex_c", "Fex_dpto", "Mes"
)

# --- Aplicar exclusiones ------------------------------------
train_h <- train_h |> select(-any_of(excluir_hogares))
test_h  <- test_h  |> select(-any_of(excluir_hogares))
train_p <- train_p |> select(-any_of(excluir_personas))
test_p  <- test_p  |> select(-any_of(excluir_personas))

# --- Limpiar nombres de columnas ----------------------------
train_h <- train_h |> janitor::clean_names()
test_h  <- test_h  |> janitor::clean_names()
train_p <- train_p |> janitor::clean_names()
test_p  <- test_p  |> janitor::clean_names()

# --- Agregar personas al nivel de hogar ---------------------
agregar_personas <- function(df_personas) {
  df_personas |>
    mutate(
      # Educación: 9 = no sabe/no informa → origen de nivel_educ_max7
      p6210 = na_if(p6210, 9),
      # Salud: 9 = no sabe/no informa
      p6090 = na_if(p6090, 9)
    ) |>
    group_by(id) |>
    summarise(
      # Demografía
      prop_mujeres  = mean(p6020 == 2, na.rm = TRUE),
      edad_promedio = mean(p6040, na.rm = TRUE),
      edad_max      = max(p6040, na.rm = TRUE),
      edad_min      = min(p6040, na.rm = TRUE),
      n_menores_18  = sum(p6040 < 18, na.rm = TRUE),
      n_mayores_65  = sum(p6040 > 65, na.rm = TRUE),
      jefe_mujer    = as.integer(any(p6050 == 1 & p6020 == 2, na.rm = TRUE)),
      
      # Educación
      nivel_educ_max = {
        val <- suppressWarnings(max(p6210, na.rm = TRUE))
        factor(ifelse(is.infinite(val), NA_real_, val), levels = 1:6)},
      n_sin_educacion   = sum(p6210 == 1, na.rm = TRUE),
      
      # Estado laboral
      n_ocupados        = sum(oc == 1, na.rm = TRUE),
      n_desocupados     = sum(des == 1, na.rm = TRUE),
      n_inactivos       = sum(ina == 1, na.rm = TRUE),
      n_pet             = sum(pet == 1, na.rm = TRUE),
      tasa_ocupacion    = sum(oc == 1, na.rm = TRUE) /
        pmax(sum(pet == 1, na.rm = TRUE), 1),
      
      # Características laborales
      prop_cuenta_propia   = mean(p6430 == 4, na.rm = TRUE),
      horas_trabajo_prom   = mean(p6800, na.rm = TRUE),
      prop_empresa_pequena = mean(p6870 %in% c(1, 2), na.rm = TRUE),
      prop_segundo_trabajo = mean(p7040 == 1, na.rm = TRUE),
      
      # Seguridad social
      prop_cotiza_pension  = mean(p6920 == 1, na.rm = TRUE),
      prop_afiliado_salud  = mean(p6090 == 1, na.rm = TRUE),
      prop_reg_subsidiado  = mean(p6100 == 3, na.rm = TRUE),
      
      # Características del jefe del hogar 
      educ_jefe = first(p6210[p6050 == 1]), 
      ocup_jefe = first(oc[p6050 == 1]), 
      edad_jefe = first(p6040[p6050 == 1]),
      
      .groups = "drop"
    )
}

cat("\n>>> Agregando personas al nivel de hogar...\n")
train_p_agg <- agregar_personas(train_p)
test_p_agg  <- agregar_personas(test_p)

# --- Join hogares + personas agregadas ----------------------
train <- train_h |>
  left_join(train_p_agg, by = c("id"))

test <- test_h |>
  left_join(test_p_agg, by = c("id"))

# --- Variables categóricas como factor ----------------------
vars_factor <- c("clase", "dominio", "depto", "p5090")

train <- train |>
  mutate(across(all_of(vars_factor), as.factor))

test <- test |>
  mutate(across(all_of(vars_factor), as.factor))


# --- Verificar join -----------------------------------------
cat("\nDims después del join:\n")
cat("  train:", dim(train), "\n")
cat("  test: ", dim(test),  "\n")

hogares_sin_personas <- sum(is.na(train$n_per))
if (hogares_sin_personas > 0) {
  cat("  AVISO:", hogares_sin_personas,
      "hogares sin match en personas\n")
}

# --- Resumen de nulos ---------------------------------------
na_resumen <- train |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  tidyr::pivot_longer(everything(),
                      names_to  = "variable",
                      values_to = "n_nulos") |>
  filter(n_nulos > 0) |>
  arrange(desc(n_nulos))

cat("\nVariables con nulos en train:\n")
print(na_resumen, n = 30)

# --- Imputación de variables laborales agregadas ------------
# NA significa que nadie en el hogar tiene esa característica → 0
vars_laborales <- c(
  "prop_cuenta_propia",
  "horas_trabajo_prom",
  "prop_segundo_trabajo",
  "prop_cotiza_pension",
  "prop_reg_subsidiado",
  "prop_empresa_pequena"
)

train <- train |>
  mutate(across(all_of(vars_laborales), ~ replace_na(., 0)))

test <- test |>
  mutate(across(all_of(vars_laborales), ~ replace_na(., 0)))

# --- Imputación NAs residuales ------------------------------
# nivel_educ_max: hogares sin info educativa → moda = 2 (primaria)
train <- train |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "2"))
test <- test |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "2"))

# educ_jefe: misma lógica que nivel_educ_max → primaria
train <- train |> mutate(educ_jefe = replace_na(educ_jefe, 2))
test  <- test  |> mutate(educ_jefe = replace_na(educ_jefe, 2))

#ocup_jefe → 0
train <- train |>
  mutate(across(c(ocup_jefe),
                ~ replace_na(., 0)))
test <- test |>
  mutate(across(c(ocup_jefe),
                ~ replace_na(., 0)))

# prop_afiliado_salud y formal_x_salud → 0
train <- train |>
  mutate(across(c(prop_afiliado_salud),
                ~ replace_na(., 0)))
test <- test |>
  mutate(across(c(prop_afiliado_salud),
                ~ replace_na(., 0)))

train$pobre <- factor(train$pobre, levels = c(0, 1), labels = c("no", "si"))

# --- Verificar stats descriptivas --------------------------------------------

skim(train |> select(-id))

# --- Winsorización de outliers extremos ---------------------
winsorizr <- function(x, p = 0.99) {
  cap <- quantile(x, p, na.rm = TRUE)
  pmin(x, cap)
}

vars_winsorizar <- c(
  "p5000",
  "p5010",
  "nper",
  "npersug",
  "n_pet",
  "n_menores_18",
  "n_ocupados",
  "n_inactivos",
  "horas_trabajo_prom"
)

train <- train |> mutate(across(all_of(vars_winsorizar), winsorizr))
test  <- test  |> mutate(across(all_of(vars_winsorizar), winsorizr))

# --- Guardar ------------------------------------------------
saveRDS(train, here(paths$processed, "train_clean.rds"))
saveRDS(test,  here(paths$processed, "test_clean.rds"))

cat("\n>>> 00_clean.R completado\n")
cat("    train:", nrow(train), "filas x", ncol(train), "columnas\n")
cat("    test: ", nrow(test),  "filas x", ncol(test),  "columnas\n")

# --- Limpiar entorno ----------------------------------------
rm(train_h, test_h, train_p, test_p, train_p_agg, test_p_agg,
   excluir_hogares, excluir_personas, vars_laborales, vars_factor,
   hogares_sin_personas, na_resumen)

gc()