# ============================================================
# 00_clean_v2.R
# Limpieza general del dataset — versión 2
#
# Cambios respecto a v1:
#   [Bug1+2] Imputación NA de nivel_educ_max y educ_jefe
#            corregida: nivel 3 (primaria) en lugar de 2 (preescolar)
#   [Bug3]   Winsorización: caps computados desde train,
#            aplicados a test (evita leakage)
#   [Bug4]   P5000/P5010 código 98 ("no sabe") → NA antes del join
#   [v2]     Variables de hogares renombradas según diccionario
#   [v2]     P6100 código 9 (NS/NR) → NA (faltaba)
#   [v2]     Conversión a factor con nuevos nombres
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
    "  Descárgalos desde la competencia.\n",
    "  Luego descomprime el zip en la misma carpeta.\n",
    "========================================================\n"
  )
} else {
  cat(">>> Datos encontrados\n")
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
  # Leakage directo: ingresos y líneas de pobreza
  "Ingtotug", "Ingtotugarr", "Ingpcug",
  "Lp", "Li",
  "Indigente", "Npobres", "Nindigentes",
  # Administrativas
  "Fex_c", "Fex_dpto", "Mes",
  # Redundante con Nper en casi todos los hogares
  "Npersug",
  # Montos de vivienda — el tipo (P5090) ya captura tenencia
  "P5100", "P5130", "P5140"
)

excluir_personas <- c(
  # Estrato — removido del test
  "Estrato1",
  # Montos de ingresos laborales (leakage)
  "P6500",
  "P6510s1", "P6510s2",
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
  # Ingresos agregados e imputados (leakage)
  "Impa", "Isa", "Ie", "Imdi",
  "Iof1", "Iof2", "Iof3h", "Iof3i", "Iof6",
  # Flags de imputación (leakage)
  "Cclasnr2", "Cclasnr3", "Cclasnr4", "Cclasnr5",
  "Cclasnr6", "Cclasnr7", "Cclasnr8", "Cclasnr11",
  # Versiones imputadas (leakage)
  "Impaes", "Isaes", "Iees", "Imdies",
  "Iof1es", "Iof2es", "Iof3hes", "Iof3ies", "Iof6es",
  # Totales de ingreso (leakage)
  "Ingtotob", "Ingtotes", "Ingtot",
  # Administrativas
  "Fex_c", "Fex_dpto", "Mes"
)

# --- Aplicar exclusiones ------------------------------------
train_h <- train_h |> select(-any_of(excluir_hogares))
test_h  <- test_h  |> select(-any_of(excluir_hogares))
train_p <- train_p |> select(-any_of(excluir_personas))
test_p  <- test_p  |> select(-any_of(excluir_personas))

# --- Limpiar nombres de columnas (lowercase) ----------------
train_h <- train_h |> janitor::clean_names()
test_h  <- test_h  |> janitor::clean_names()
train_p <- train_p |> janitor::clean_names()
test_p  <- test_p  |> janitor::clean_names()

# --- [Bug4] NS/NR en hogares → NA ---------------------------
# p5000/p5010: código 98 = "no sabe", 99 = "no informa"
# Se convierten a NA e imputan con mediana de train
train_h <- train_h |>
  mutate(
    p5000 = na_if(p5000, 98L) |> na_if(99L),
    p5010 = na_if(p5010, 98L) |> na_if(99L)
  )
test_h <- test_h |>
  mutate(
    p5000 = na_if(p5000, 98L) |> na_if(99L),
    p5010 = na_if(p5010, 98L) |> na_if(99L)
  )

# Imputar con mediana de train (calculada antes del join)
med_p5000 <- median(train_h$p5000, na.rm = TRUE)
med_p5010 <- median(train_h$p5010, na.rm = TRUE)

train_h <- train_h |>
  mutate(
    p5000 = replace_na(p5000, med_p5000),
    p5010 = replace_na(p5010, med_p5010)
  )
test_h <- test_h |>
  mutate(
    p5000 = replace_na(p5000, med_p5000),
    p5010 = replace_na(p5010, med_p5010)
  )

# --- Agregar personas al nivel de hogar ---------------------
agregar_personas <- function(df_personas) {
  df_personas |>
    mutate(
      # Educación: 9 = NS/NR → NA
      p6210 = na_if(p6210, 9),
      # Afiliación salud: 9 = NS/NR → NA
      p6090 = na_if(p6090, 9),
      # [v2] Régimen salud: 9 = NS/NR → NA (faltaba en v1)
      p6100 = na_if(p6100, 9)
    ) |>
    group_by(id) |>
    summarise(
      # --- Demografía -----------------------------------------
      prop_mujeres  = mean(p6020 == 2, na.rm = TRUE),
      edad_promedio = mean(p6040, na.rm = TRUE),
      edad_max      = max(p6040, na.rm = TRUE),
      edad_min      = min(p6040, na.rm = TRUE),
      n_menores_18  = sum(p6040 < 18, na.rm = TRUE),
      n_mayores_65  = sum(p6040 > 65, na.rm = TRUE),
      jefe_mujer    = as.integer(any(p6050 == 1 & p6020 == 2, na.rm = TRUE)),

      # --- Educación ------------------------------------------
      # nivel_educ_max: factor nominal 1=ninguno … 6=superior
      nivel_educ_max = {
        val <- suppressWarnings(max(p6210, na.rm = TRUE))
        factor(ifelse(is.infinite(val), NA_real_, val), levels = 1:6)
      },
      n_sin_educacion = sum(p6210 == 1, na.rm = TRUE),

      # --- Estado laboral -------------------------------------
      n_ocupados     = sum(oc == 1, na.rm = TRUE),
      n_desocupados  = sum(des == 1, na.rm = TRUE),
      n_inactivos    = sum(ina == 1, na.rm = TRUE),
      n_pet          = sum(pet == 1, na.rm = TRUE),
      tasa_ocupacion = sum(oc == 1, na.rm = TRUE) /
        pmax(sum(pet == 1, na.rm = TRUE), 1),

      # --- Características laborales --------------------------
      prop_cuenta_propia   = mean(p6430 == 4, na.rm = TRUE),
      horas_trabajo_prom   = mean(p6800, na.rm = TRUE),
      prop_empresa_pequena = mean(p6870 %in% c(1, 2), na.rm = TRUE),
      prop_segundo_trabajo = mean(p7040 == 1, na.rm = TRUE),

      # --- Seguridad social -----------------------------------
      prop_cotiza_pension  = mean(p6920 == 1, na.rm = TRUE),
      prop_afiliado_salud  = mean(p6090 == 1, na.rm = TRUE),
      prop_reg_subsidiado  = mean(p6100 == 3, na.rm = TRUE),

      # --- Características del jefe del hogar -----------------
      educ_jefe    = first(p6210[p6050 == 1]),
      ocup_jefe    = first(oc[p6050 == 1]),
      edad_jefe    = first(p6040[p6050 == 1]),
      # Oficio del jefe: código CNO 2 dígitos (0-99)
      # NA cuando el jefe no está empleado — se recodifica post-join
      oficio_jefe_raw = first(oficio[p6050 == 1]),

      .groups = "drop"
    )
}

cat("\n>>> Agregando personas al nivel de hogar...\n")
train_p_agg <- agregar_personas(train_p)
test_p_agg  <- agregar_personas(test_p)

# --- Join hogares + personas agregadas ----------------------
train <- train_h |> left_join(train_p_agg, by = "id")
test  <- test_h  |> left_join(test_p_agg,  by = "id")

# --- [v2] Renombrar variables hogares según diccionario -----
# Mapeo: nombre_limpio_original → nombre_descriptivo
renombrar <- c(
  clase  = "cabecera",
  depto  = "departamento",
  p5000  = "cuartos_hogar",
  p5010  = "cuartos_ocupados",
  p5090  = "tipo_ocupacion_vivienda",
  nper   = "n_personas"
  # dominio → dominio (sin cambio)
)

train <- train |> rename(any_of(renombrar))
test  <- test  |> rename(any_of(renombrar))

# --- [v2] Recodificar oficio_jefe → grupo CNO 1 dígito -----
# CNO 2-digit // 10 da el grupo mayor (0-9):
#   0=militar, 1=directivos, 2=profesionales, 3=técnicos,
#   4=oficina, 5=servicios/ventas, 6=agropecuario,
#   7=artesanos, 8=operadores, 9=elementales
# Jefes no empleados (NA) → nivel "sin_ocup" (alta tasa de pobreza)
recodificar_oficio <- function(df) {
  df |>
    mutate(
      oficio_jefe_grp = case_when(
        is.na(oficio_jefe_raw)              ~ "sin_ocup",
        TRUE ~ as.character(oficio_jefe_raw %/% 10L)
      ),
      oficio_jefe_grp = factor(oficio_jefe_grp,
                               levels = c(as.character(0:9), "sin_ocup"))
    ) |>
    select(-oficio_jefe_raw)
}

train <- recodificar_oficio(train)
test  <- recodificar_oficio(test)

# --- Verificar join -----------------------------------------
cat("\nDims después del join:\n")
cat("  train:", dim(train), "\n")
cat("  test: ", dim(test),  "\n")

hogares_sin_personas <- sum(is.na(train$n_personas))
if (hogares_sin_personas > 0) {
  cat("  AVISO:", hogares_sin_personas, "hogares sin match en personas\n")
}

# --- [v2] Variables categóricas → factor --------------------
# Nominales de hogares
vars_factor_nominal <- c(
  "cabecera",               # 1=Urbano, 2=Rural
  "dominio",                # 25 áreas/dominios
  "departamento",           # 24 departamentos
  "tipo_ocupacion_vivienda" # 1=propia pagada … 6=otra
)

# nivel_educ_max ya es factor (creado en agregar_personas)
# Se convierte con factores el resto:
train <- train |> mutate(across(all_of(vars_factor_nominal), as.factor))
test  <- test  |> mutate(across(all_of(vars_factor_nominal), as.factor))

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

# --- Imputación de proporciones laborales -------------------
# NA en proporciones laborales = nadie en el hogar tiene esa característica → 0
vars_laborales <- c(
  "prop_cuenta_propia",
  "horas_trabajo_prom",
  "prop_segundo_trabajo",
  "prop_cotiza_pension",
  "prop_reg_subsidiado",
  "prop_empresa_pequena"
)

train <- train |> mutate(across(all_of(vars_laborales), ~ replace_na(., 0)))
test  <- test  |> mutate(across(all_of(vars_laborales), ~ replace_na(., 0)))

# --- Imputación NAs residuales ------------------------------
# [Bug1+2] nivel_educ_max NA → nivel 3 (primaria básica), no 2 (preescolar)
train <- train |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "3"))
test <- test |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "3"))

# educ_jefe NA → primaria básica (3)
train <- train |> mutate(educ_jefe = replace_na(educ_jefe, 3))
test  <- test  |> mutate(educ_jefe = replace_na(educ_jefe, 3))

# ocup_jefe NA → 0 (jefe no ocupado)
train <- train |> mutate(ocup_jefe = replace_na(ocup_jefe, 0))
test  <- test  |> mutate(ocup_jefe = replace_na(ocup_jefe, 0))

# prop_afiliado_salud NA → 0
train <- train |> mutate(prop_afiliado_salud = replace_na(prop_afiliado_salud, 0))
test  <- test  |> mutate(prop_afiliado_salud = replace_na(prop_afiliado_salud, 0))

# --- Stats descriptivas -------------------------------------
skim(train |> select(-id))

# --- [Bug3] Winsorización con caps de TRAIN -----------------
# Caps computados SOLO desde train; aplicados a ambos para evitar leakage
vars_winsorizar <- c(
  "cuartos_hogar",    # ex p5000
  "cuartos_ocupados", # ex p5010
  "n_personas",       # ex nper
  "n_pet",
  "n_menores_18",
  "n_ocupados",
  "n_inactivos",
  "horas_trabajo_prom"
)

caps_winsor <- train |>
  summarise(across(all_of(vars_winsorizar),
                   ~ quantile(.x, 0.99, na.rm = TRUE)))

cat("\nCaps de winsorización (p99 de train):\n")
print(caps_winsor)

for (v in vars_winsorizar) {
  cap <- caps_winsor[[v]]
  train <- train |> mutate(!!v := pmin(.data[[v]], cap))
  test  <- test  |> mutate(!!v := pmin(.data[[v]], cap))
}

# --- Guardar ------------------------------------------------
saveRDS(train, here(paths$processed, "train_clean.rds"))
saveRDS(test,  here(paths$processed, "test_clean.rds"))

cat("\n>>> 00_clean_v2.R completado\n")
cat("    train:", nrow(train), "filas x", ncol(train), "columnas\n")
cat("    test: ", nrow(test),  "filas x", ncol(test),  "columnas\n")

# --- Limpiar entorno ----------------------------------------
rm(train_h, test_h, train_p, test_p, train_p_agg, test_p_agg,
   excluir_hogares, excluir_personas, vars_laborales,
   vars_factor_nominal, hogares_sin_personas, na_resumen,
   caps_winsor, vars_winsorizar, renombrar, med_p5000, med_p5010,
   recodificar_oficio)

gc()
