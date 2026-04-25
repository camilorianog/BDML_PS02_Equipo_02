# ============================================================
# 00_clean_v2.R
# Limpieza REDUCIDA — base minimalista (núcleo simple)
# ------------------------------------------------------------
# Criterio de selección:
#   · Variables con <10% NA en train
#   · Núcleo simple de agregaciones (demografía, educación,
#     ocupación jefe, ocupación agregada)
#   · Se excluyen proporciones laborales y de salud por alta
#     tasa de missings en hogares sin ocupados
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

# --- Limpiar nombres ----------------------------------------
train_h <- train_h |> janitor::clean_names()
test_h  <- test_h  |> janitor::clean_names()
train_p <- train_p |> janitor::clean_names()
test_p  <- test_p  |> janitor::clean_names()

# --- Variables de hogar que SE CONSERVAN (<10% NA) ----------
# id + Clase + Dominio + Depto + P5000 + P5010 + P5090 + Nper
# (+ pobre en train)
cols_h_train <- c("id", "clase", "dominio", "depto",
                  "p5000", "p5010", "p5090", "nper", "pobre")
cols_h_test  <- c("id", "clase", "dominio", "depto",
                  "p5000", "p5010", "p5090", "nper")

train_h <- train_h |> select(any_of(cols_h_train))
test_h  <- test_h  |> select(any_of(cols_h_test))

# --- NS/NR en hogares → NA → mediana de TRAIN --------------
# p5000/p5010: 98 = "no sabe", 99 = "no informa"
train_h <- train_h |>
  mutate(p5000 = na_if(p5000, 98L) |> na_if(99L),
         p5010 = na_if(p5010, 98L) |> na_if(99L))
test_h <- test_h |>
  mutate(p5000 = na_if(p5000, 98L) |> na_if(99L),
         p5010 = na_if(p5010, 98L) |> na_if(99L))

med_p5000 <- median(train_h$p5000, na.rm = TRUE)
med_p5010 <- median(train_h$p5010, na.rm = TRUE)

train_h <- train_h |>
  mutate(p5000 = replace_na(p5000, med_p5000),
         p5010 = replace_na(p5010, med_p5010))
test_h <- test_h |>
  mutate(p5000 = replace_na(p5000, med_p5000),
         p5010 = replace_na(p5010, med_p5010))

# --- Agregar personas al nivel de hogar (núcleo simple) -----
# Se mantienen solo agregaciones con cobertura alta (<10% NA):
#   · Demografía (sexo, edad, jefatura)  → 0% NA
#   · Educación hogar y jefe (p6210)     → <5% NA
#   · Conteos laborales (oc, pet)        → 0% NA
#   · Características básicas del jefe   → <5% NA
# Se EXCLUYEN proporciones laborales (prop_cuenta_propia,
# horas_trabajo_prom, prop_cotiza_pension, prop_afiliado_salud,
# prop_reg_subsidiado, prop_empresa_pequena, prop_segundo_trabajo)
# porque son NA para hogares sin ocupados, lo que introduce
# imputación artificial en >15% de hogares.
agregar_personas <- function(df_personas) {

  get_mode <- function(x) {
    if (all(is.na(x))) return(NA_integer_)
    as.integer(names(table(x))[which.max(table(x))])
  }

  df_personas |>
    mutate(p6210 = na_if(p6210, 9)) |>
    group_by(id) |>
    mutate(p6210 = coalesce(p6210, get_mode(p6210))) |>
    summarise(
      # --- Demografía ---------------------------------------
      prop_mujeres  = mean(p6020 == 2,  na.rm = TRUE),
      edad_promedio = mean(p6040,       na.rm = TRUE),
      n_menores_18  = sum(p6040 < 18,   na.rm = TRUE),
      n_mayores_65  = sum(p6040 > 65,   na.rm = TRUE),
      jefe_mujer    = as.integer(any(p6050 == 1 & p6020 == 2,
                                     na.rm = TRUE)),

      # --- Educación (hogar) --------------------------------
      # nivel_educ_max: factor nominal 1..6 (no ordenado)
      # 1=Ninguno, 2=Preescolar, 3=Primaria, 4=Secundaria,
      # 5=Media, 6=Superior
      nivel_educ_max = {
        val <- suppressWarnings(max(p6210, na.rm = TRUE))
        factor(ifelse(is.infinite(val), NA_real_, val),
               levels = 1:6)
      },

      # --- Ocupación (conteos, sin NAs) ---------------------
      n_ocupados     = sum(oc  == 1, na.rm = TRUE),
      n_pet          = sum(pet == 1, na.rm = TRUE),
      tasa_ocupacion = sum(oc == 1, na.rm = TRUE) /
                       pmax(sum(pet == 1, na.rm = TRUE), 1),

      # --- Jefe del hogar -----------------------------------
      educ_jefe = first(p6210[p6050 == 1]),
      ocup_jefe = first(oc[p6050 == 1]),
      edad_jefe = first(p6040[p6050 == 1]),

      .groups = "drop"
    )
}

cat("\n>>> Agregando personas al nivel de hogar...\n")
train_p_agg <- agregar_personas(train_p)
test_p_agg  <- agregar_personas(test_p)

# --- Join ---------------------------------------------------
train <- train_h |> left_join(train_p_agg, by = "id")
test  <- test_h  |> left_join(test_p_agg,  by = "id")

# --- Renombrar (formato: nuevo = viejo) ---------------------
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

# --- Verificar join -----------------------------------------
cat("\nDims después del join:\n")
cat("  train:", nrow(train), "x", ncol(train), "\n")
cat("  test: ", nrow(test),  "x", ncol(test),  "\n")

# --- Categóricas nominales → factor -------------------------
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

# --- Imputación NAs residuales ------------------------------
# nivel_educ_max NA → "3" (primaria, moda confirmada)
train <- train |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "3"))
test <- test |>
  mutate(nivel_educ_max = fct_na_value_to_level(nivel_educ_max, level = "3"))

# educ_jefe NA → 3 (moda)
train <- train |> mutate(educ_jefe = replace_na(educ_jefe, 3))
test  <- test  |> mutate(educ_jefe = replace_na(educ_jefe, 3))

# ocup_jefe NA → 0 (jefe no ocupado)
train <- train |> mutate(ocup_jefe = replace_na(ocup_jefe, 0))
test  <- test  |> mutate(ocup_jefe = replace_na(ocup_jefe, 0))

# edad_jefe NA → mediana de train
mediana_edad_jefe <- median(train$edad_jefe, na.rm = TRUE)
train <- train |> mutate(edad_jefe = replace_na(edad_jefe, mediana_edad_jefe))
test  <- test  |> mutate(edad_jefe = replace_na(edad_jefe, mediana_edad_jefe))

# --- Stats descriptivas -------------------------------------
skim(train |> select(-id))

# --- Winsorización (caps de TRAIN) --------------------------
vars_winsorizar <- c(
  "cuartos_hogar",
  "cuartos_ocupados",
  "n_personas",
  "n_pet",
  "n_menores_18",
  "n_ocupados"
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

cat("\n>>> 00_clean_v2.R (reducido) completado\n")
cat("    train:", nrow(train), "x", ncol(train), "\n")
cat("    test: ", nrow(test),  "x", ncol(test),  "\n")
cat("\n    Variables del núcleo simple:\n")
cat("    · Hogar       : cabecera, dominio, departamento,\n")
cat("                    cuartos_hogar, cuartos_ocupados,\n")
cat("                    tipo_ocupacion_vivienda, n_personas\n")
cat("    · Demografía  : prop_mujeres, edad_promedio,\n")
cat("                    n_menores_18, n_mayores_65, jefe_mujer\n")
cat("    · Educación   : nivel_educ_max, educ_jefe\n")
cat("    · Ocupación   : n_ocupados, n_pet, tasa_ocupacion, ocup_jefe\n")
cat("    · Jefe        : edad_jefe\n")

# --- Limpiar entorno ----------------------------------------
rm(train_h, test_h, train_p, test_p, train_p_agg, test_p_agg,
   cols_h_train, cols_h_test, vars_factor_nominal,
   na_resumen, caps_winsor, vars_winsorizar,
   med_p5000, med_p5010, mediana_edad_jefe,
   archivos_necesarios, faltantes, agregar_personas)

gc()
