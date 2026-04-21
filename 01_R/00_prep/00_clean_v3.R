# ============================================================
# 00_clean_v3.R
#
# Cambios respecto a v3:
#   [1] Formalidad laboral:
#           prop_formal  = proporción de ocupados con primas en el hogar
#           jefe_formal  = dummy si el jefe recibió alguna prima
#                   ( P6630s1 prima servicios, P6630s2 prima navidad,
#                    P6630s3 prima vacaciones — flags de empleo formal)
#   [2] Estabilidad laboral del jefe:
#           antiguedad_jefe = meses de permanencia en empresa (P6426)
#           NA cuando el jefe no está empleado → 0 post-join
#   [P3] Subempleo:
#           tasa_subempleo = proporción del hogar que quiere + buscó
#                            + estaba disponible para más horas
#                            (P7090 & P7110 & P7120 == 1)
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
cat("  train_hogares: ", nrow(train_h), "x", ncol(train_h), "\n")
cat("  test_hogares:  ", nrow(test_h),  "x", ncol(test_h),  "\n")
cat("  train_personas:", nrow(train_p), "x", ncol(train_p), "\n")
cat("  test_personas: ", nrow(test_p),  "x", ncol(test_p),  "\n")

# --- Variables a excluir ------------------------------------

excluir_hogares <- c(
  # Leakage directo: ingresos y líneas de pobreza
  "Ingtotug", "Ingtotugarr", "Ingpcug",
  "Lp", "Li",
  "Indigente", "Npobres", "Nindigentes",
  # Administrativas
  "Fex_c", "Fex_dpto", "Mes",
  # Redundante con Nper
  "Npersug",
  # Montos de vivienda — tipo (P5090) ya captura tenencia
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

# --- Limpiar nombres de columnas ----------------------------
train_h <- train_h |> janitor::clean_names()
test_h  <- test_h  |> janitor::clean_names()
train_p <- train_p |> janitor::clean_names()
test_p  <- test_p  |> janitor::clean_names()

# --- NS/NR en hogares → NA ----------------------------------
# p5000/p5010: 98 = "no sabe", 99 = "no informa"
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
      # NS/NR → NA en variables de educación y salud
      p6210 = na_if(p6210, 9),
      p6090 = na_if(p6090, 9),
      p6100 = na_if(p6100, 9),
      # NS/NR → NA en variables laborales
      p6430 = na_if(p6430, 9),
      p6800 = na_if(p6800, 99),
      p6870 = na_if(p6870, 9),
      p6920 = na_if(p6920, 9),
      p7040 = na_if(p7040, 9),
      # [PasoA] NS/NR en primas → NA
      p6630s1 = na_if(p6630s1, 9),
      p6630s2 = na_if(p6630s2, 9),
      p6630s3 = na_if(p6630s3, 9),
      # [PasoC] NS/NR en subempleo → NA
      p7090 = na_if(p7090, 9),
      p7110 = na_if(p7110, 9),
      p7120 = na_if(p7120, 9)
    ) |>
    group_by(id) |>
    summarise(
      
      # --- Demografía -----------------------------------------
      prop_mujeres  = mean(p6020 == 2,  na.rm = TRUE),
      edad_promedio = mean(p6040,        na.rm = TRUE),
      edad_max      = max(p6040,         na.rm = TRUE),
      edad_min      = min(p6040,         na.rm = TRUE),
      n_menores_18  = sum(p6040 < 18,    na.rm = TRUE),
      n_mayores_65  = sum(p6040 > 65,    na.rm = TRUE),
      jefe_mujer    = as.integer(any(p6050 == 1 & p6020 == 2, na.rm = TRUE)),
      
      # --- Educación ------------------------------------------
      # [BugFix] ordered = TRUE: escala real 1 < 2 < 3 < 4 < 5 < 6
      # 1=Ninguno, 2=Preescolar, 3=Primaria, 4=Secundaria,
      # 5=Media, 6=Superior/universitaria (incl. técnico y tecnólogo)
      nivel_educ_max = {
        val <- suppressWarnings(max(p6210, na.rm = TRUE))
        factor(
          ifelse(is.infinite(val), NA_real_, val),
          levels  = 1:6,
          ordered = TRUE   # [BugFix] ordenado
        )
      },
      n_sin_educacion = sum(p6210 == 1, na.rm = TRUE),
      
      # --- Estado laboral -------------------------------------
      n_ocupados     = sum(oc == 1,  na.rm = TRUE),
      n_desocupados  = sum(des == 1, na.rm = TRUE),
      n_inactivos    = sum(ina == 1, na.rm = TRUE),
      n_pet          = sum(pet == 1, na.rm = TRUE),
      tasa_ocupacion = sum(oc == 1, na.rm = TRUE) /
        pmax(sum(pet == 1, na.rm = TRUE), 1),
      
      # --- Características laborales --------------------------
      prop_cuenta_propia   = mean(p6430 == 4,        na.rm = TRUE),
      horas_trabajo_prom   = mean(p6800,             na.rm = TRUE),
      prop_empresa_pequena = mean(p6870 %in% c(1,2), na.rm = TRUE),
      prop_segundo_trabajo = mean(p7040 == 1,        na.rm = TRUE),
      
      # --- Seguridad social -----------------------------------
      prop_cotiza_pension = mean(p6920 == 1, na.rm = TRUE),
      prop_afiliado_salud = mean(p6090 == 1, na.rm = TRUE),
      prop_reg_subsidiado = mean(p6100 == 3, na.rm = TRUE),
      
      # --- [PasoA] Formalidad laboral -------------------------
      # Formal = recibió prima de servicios, navidad o vacaciones
      # Restringido a ocupados para no distorsionar la proporción
      prop_formal = mean(
        (p6630s1 == 1 | p6630s2 == 1 | p6630s3 == 1) & oc == 1,
        na.rm = TRUE
      ),
      jefe_formal = as.integer(first(
        (p6630s1[p6050 == 1] == 1 |
           p6630s2[p6050 == 1] == 1 |
           p6630s3[p6050 == 1] == 1),
        default = FALSE
      )),
      
      # --- [PasoB] Estabilidad laboral del jefe ---------------
      # Meses continuos en la misma empresa (NA si no empleado → 0)
      antiguedad_jefe = first(p6426[p6050 == 1], default = NA_real_),
      
      # --- [PasoC] Subempleo visible --------------------------
      # Quiere + buscó + disponible para más horas
      tasa_subempleo = mean(
        p7090 == 1 & p7110 == 1 & p7120 == 1,
        na.rm = TRUE
      ),
      
      # --- Características del jefe ---------------------------
      educ_jefe       = first(p6210[p6050 == 1]),
      ocup_jefe       = first(oc[p6050 == 1]),
      edad_jefe       = first(p6040[p6050 == 1]),
      sexo_jefe       = first(p6020[p6050 == 1]),
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

# --- [BugFix] Renombrar: formato correcto nuevo = viejo -----
train <- train |>
  rename(
    cabecera                = clase,
    departamento            = depto,
    cuartos_hogar           = p5000,
    cuartos_ocupados        = p5010,
    tipo_ocupacion_vivienda = p5090,
    n_personas              = nper
  )

test <- test |>
  rename(
    cabecera                = clase,
    departamento            = depto,
    cuartos_hogar           = p5000,
    cuartos_ocupados        = p5010,
    tipo_ocupacion_vivienda = p5090,
    n_personas              = nper
  )

# --- Recodificar oficio_jefe → grupo CNO 1 dígito ----------
# 0=militares, 1=directivos, 2=profesionales, 3=técnicos,
# 4=oficina, 5=servicios/ventas, 6=agropecuario,
# 7=artesanos, 8=operadores, 9=elementales, sin_ocup=no empleado
recodificar_oficio <- function(df) {
  df |>
    mutate(
      oficio_jefe_grp = case_when(
        is.na(oficio_jefe_raw) ~ "sin_ocup",
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
cat("  train:", nrow(train), "x", ncol(train), "\n")
cat("  test: ", nrow(test),  "x", ncol(test),  "\n")

hogares_sin_personas <- sum(is.na(train$n_personas))
if (hogares_sin_personas > 0) {
  cat("  AVISO:", hogares_sin_personas, "hogares sin match en personas\n")
}

# --- Variables categóricas nominales → factor ---------------
vars_factor_nominal <- c(
  "cabecera",
  "dominio",
  "departamento",
  "tipo_ocupacion_vivienda"
)

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

# --- Imputación → 0 -----------------------------------------
# NA en proporciones = nadie en el hogar tiene esa característica
vars_imp_cero <- c(
  "prop_cuenta_propia",
  "horas_trabajo_prom",
  "prop_segundo_trabajo",
  "prop_cotiza_pension",
  "prop_reg_subsidiado",
  "prop_empresa_pequena",
  "prop_afiliado_salud",
  "ocup_jefe",
  "prop_formal",       # [PasoA] sin ocupados → 0
  "jefe_formal",       # [PasoA] jefe no empleado → 0
  "tasa_subempleo"     # [PasoC] sin PET → 0
)

train <- train |> mutate(across(all_of(vars_imp_cero), ~ replace_na(., 0)))
test  <- test  |> mutate(across(all_of(vars_imp_cero), ~ replace_na(., 0)))

# --- Imputación NAs residuales ------------------------------

# [BugFix] nivel_educ_max → moda confirmada = 3 (primaria, 134k obs)
# nivel 6 tiene 127k — primaria es la moda real
train <- train |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "3"))
test <- test |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "3"))

# educ_jefe → moda = 3 (primaria)
train <- train |> mutate(educ_jefe = replace_na(educ_jefe, 3))
test  <- test  |> mutate(educ_jefe = replace_na(educ_jefe, 3))

# edad_jefe → mediana del train
# sexo_jefe → moda del train
mediana_edad_jefe <- median(train$edad_jefe, na.rm = TRUE)
moda_sexo_jefe    <- as.integer(
  names(sort(table(train$sexo_jefe), decreasing = TRUE)[1])
)

train <- train |>
  mutate(
    edad_jefe = replace_na(edad_jefe, mediana_edad_jefe),
    sexo_jefe = replace_na(sexo_jefe, moda_sexo_jefe)
  )
test <- test |>
  mutate(
    edad_jefe = replace_na(edad_jefe, mediana_edad_jefe),
    sexo_jefe = replace_na(sexo_jefe, moda_sexo_jefe)
  )

# [PasoB] antiguedad_jefe → 0 (jefe sin empleo = 0 meses)
train <- train |> mutate(antiguedad_jefe = replace_na(antiguedad_jefe, 0))
test  <- test  |> mutate(antiguedad_jefe = replace_na(antiguedad_jefe, 0))

# --- Stats descriptivas -------------------------------------
skim(train |> select(-id))

# --- Winsorización con caps de TRAIN ------------------------
# Caps calculados solo en train y aplicados a test (evita leakage)
vars_winsorizar <- c(
  "cuartos_hogar",
  "cuartos_ocupados",
  "n_personas",
  "n_pet",
  "n_menores_18",
  "n_ocupados",
  "n_inactivos",
  "horas_trabajo_prom",
  "antiguedad_jefe"    # [PasoB]
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

cat("\n>>> 00_clean_v5.R completado\n")
cat("    train:", nrow(train), "x", ncol(train), "\n")
cat("    test: ", nrow(test),  "x", ncol(test),  "\n")
cat("\n    Variables en el dataset:\n")
cat("    Factores ordenados : nivel_educ_max\n")
cat("    Factores nominales : cabecera, dominio, departamento,\n")
cat("                         tipo_ocupacion_vivienda, oficio_jefe_grp\n")
cat("    Nuevas [PasoA]     : prop_formal, jefe_formal\n")
cat("    Nuevas [PasoB]     : antiguedad_jefe\n")
cat("    Nuevas [PasoC]     : tasa_subempleo\n")

# --- Limpiar entorno ----------------------------------------
rm(train_h, test_h, train_p, test_p, train_p_agg, test_p_agg,
   excluir_hogares, excluir_personas, vars_imp_cero,
   vars_factor_nominal, hogares_sin_personas, na_resumen,
   caps_winsor, vars_winsorizar,
   med_p5000, med_p5010, mediana_edad_jefe, moda_sexo_jefe,
   recodificar_oficio)

gc()