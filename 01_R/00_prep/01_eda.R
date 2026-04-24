# ==============================================================================
# 04_eda.R — Estadísticas Descriptivas Completas
# ==============================================================================
# Problem Set 2: Predicción de Pobreza — MECA 4107
# Universidad de los Andes — 2026-10
#
# Combina:
#   [CONSOLA] Tablas numéricas: target, missings, correlaciones, outliers
#   [FIGURAS] Gráficos descriptivos de las variables
#
# Input:  train_features.rds (post feature engineering)
# Output: figuras en paths$figures, tablas en paths$tables
# ==============================================================================


cat("\n>>> [EDA] Iniciando análisis descriptivo completo...\n")

# --- Cargar datos -----------------------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))
test  <- readRDS(here(paths$processed, "test_features.rds"))

cat(sprintf("   train: %d obs × %d vars\n", nrow(train), ncol(train)))
cat(sprintf("   test:  %d obs × %d vars\n", nrow(test),  ncol(test)))

# --- Paths de output --------------------------------------------------------
dir.create(here(paths$figures), recursive = TRUE, showWarnings = FALSE)
dir.create(here(paths$tables),  recursive = TRUE, showWarnings = FALSE)

# --- Función guardar figuras ------------------------------------------------
guardar_fig <- function(p, nombre, width = 12, height = 7, dpi = 300) {
  ruta <- here(paths$figures, paste0(nombre, ".png"))
  ggsave(ruta, plot = p, width = width, height = height, dpi = dpi, bg = "white")
  cat(sprintf("   Figura guardada: %s\n", basename(ruta)))
}

# --- Tema parejo todo -------------------------------------------------------
tema_pub <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, color = "#0A2240"),
    plot.subtitle    = element_text(size = 11, color = "#555555"),
    plot.caption     = element_text(size = 9,  color = "#888888", hjust = 0),
    axis.title       = element_text(size = 11, color = "#333333"),
    axis.text        = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#EEEEEE"),
    strip.text       = element_text(face = "bold", size = 11),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold")
  )

COLORES_CLASE <- c("No pobre" = "#1565C0", "Pobre" = "#B71C1C")

# ==============================================================================
# BLOQUE A — TARGET
# ==============================================================================

cat("\n=== TARGET: Pobre ===\n")
train |>
  count(pobre) |>
  mutate(
    clase = ifelse(pobre == 1, "Pobre", "No pobre"),
    pct   = round(n / sum(n) * 100, 2)
  ) |>
  select(clase, n, pct) |>
  print()

# ==============================================================================
# BLOQUE B — MISSING VALUES
# ==============================================================================

cat("\n=== MISSING VALUES (train) ===\n")
missings_train <- train |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "variable", values_to = "n_na") |>
  mutate(pct_na = round(n_na / nrow(train) * 100, 2)) |>
  filter(n_na > 0) |>
  arrange(desc(pct_na))

cat(sprintf("   Variables con missings: %d\n", nrow(missings_train)))
print(missings_train, n = 50)

# ==============================================================================
# BLOQUE D — RESUMEN NUMÉRICO (skimr)
# ==============================================================================

cat("\n=== RESUMEN NUMÉRICO (skimr) ===\n")
vars_num <- train |>
  select(-id, -pobre, -any_of(vars_cat)) |>
  select(where(is.numeric)) |>
  names()

cat(sprintf("   Variables numéricas: %d\n", length(vars_num)))
train |> select(all_of(vars_num)) |> skim() |> print()

# ==============================================================================
# BLOQUE E — MEDIAS POR GRUPO + DIFERENCIAS
# ==============================================================================

cat("\n=== MEDIAS POR GRUPO (pobre=0 vs pobre=1) ===\n")
train |>
  group_by(pobre) |>
  summarise(across(all_of(vars_num), ~ round(mean(.x, na.rm = TRUE), 3))) |>
  pivot_longer(-pobre, names_to = "variable", values_to = "media") |>
  pivot_wider(names_from = pobre, values_from = media, names_prefix = "pobre_") |>
  mutate(
    dif_abs = round(abs(pobre_1 - pobre_0), 4),
    ratio   = round(pobre_1 / pmax(pobre_0, 1e-6), 3)
  ) |>
  arrange(desc(dif_abs)) |>
  print(n = 50)

# ==============================================================================
# BLOQUE F — CORRELACIÓN CON TARGET
# ==============================================================================

cat("\n=== CORRELACIÓN CON pobre (Pearson) ===\n")
correlaciones <- train |>
  select(pobre, all_of(vars_num)) |>
  mutate(pobre = as.numeric(pobre)) |>
  cor(use = "pairwise.complete.obs") |>
  as.data.frame() |>
  rownames_to_column("variable") |>
  select(variable, pobre) |>
  filter(variable != "pobre") |>
  arrange(desc(abs(pobre))) |>
  mutate(pobre = round(pobre, 4))

print(as.data.frame(correlaciones), row.names = FALSE)

cat("\n   Top 5 correlaciones POSITIVAS con pobreza:\n")
correlaciones |> filter(pobre > 0) |> head(5) |> print()

cat("\n   Top 5 correlaciones NEGATIVAS con pobreza:\n")
correlaciones |> filter(pobre < 0) |> head(5) |> print()

# ==============================================================================
# BLOQUE G — OUTLIERS: PERCENTILES 95-99-MAX
# ==============================================================================

cat("\n=== OUTLIERS — PERCENTILES CLAVE ===\n")
train |>
  select(all_of(vars_num)) |>
  summarise(across(everything(), list(
    p95 = ~ quantile(.x, 0.95, na.rm = TRUE),
    p99 = ~ quantile(.x, 0.99, na.rm = TRUE),
    max = ~ max(.x, na.rm = TRUE)
  ))) |>
  pivot_longer(everything(),
               names_to  = c("variable", ".value"),
               names_sep = "_(?=[^_]+$)") |>
  arrange(desc(max)) |>
  print(n = 60)

# ==============================================================================
# BLOQUE H — CONSISTENCIA TRAIN vs TEST
# ==============================================================================

cat("\n=== COLUMNAS EN TRAIN PERO NO EN TEST ===\n")
print(setdiff(names(train), c(names(test), "pobre")))

cat("\n=== COLUMNAS EN TEST PERO NO EN TRAIN ===\n")
print(setdiff(names(test), names(train)))

# ==============================================================================
# BLOQUE I — REDUNDANCIA nper vs npersug
# ==============================================================================

if (all(c("nper", "npersug") %in% names(train))) {
  cat(sprintf("\n=== CORR(nper, npersug) = %.4f ===\n",
              cor(train$nper, train$npersug, use = "complete.obs")))
  cat("   Diferencia media:",
      round(mean(train$nper - train$npersug, na.rm = TRUE), 4), "\n")
  cat("   % donde nper != npersug:",
      round(mean(train$nper != train$npersug, na.rm = TRUE) * 100, 2), "%\n")
}

# ==============================================================================
# BLOQUE J — CHECK P5000 VALORES EXTREMOS (raw)
# ==============================================================================

ruta_raw_hog <- here(paths$raw, "train_hogares.csv")
if (file.exists(ruta_raw_hog)) {
  cat("\n=== P5000 VALORES EXTREMOS EN RAW ===\n")
  train_h_raw <- read.csv(ruta_raw_hog)
  cat("   P5000 == 98:", sum(train_h_raw$P5000 == 98, na.rm = TRUE), "\n")
  cat("   P5000 == 99:", sum(train_h_raw$P5000 == 99, na.rm = TRUE), "\n")
  cat("   P5000 > 20: ", sum(train_h_raw$P5000 >  20, na.rm = TRUE), "\n")
  rm(train_h_raw)
}

# ==============================================================================
# ==============================================================================
# FIGURAS
# ==============================================================================
# ==============================================================================

cat("\n\n>>> [FIGURAS] Generando gráficos...\n")

# ------------------------------------------------------------------------------
# FIGURA 1 — Desbalance de clases
# ------------------------------------------------------------------------------

cat("\n[Fig 1/6] Desbalance de clases...\n")

p1 <- train |>
  count(pobre) |>
  mutate(clase = ifelse(pobre == 1, "Pobre", "No pobre"),
         pct   = n / sum(n) * 100) |>
  ggplot(aes(x = clase, y = pct, fill = clase)) +
  geom_col(width = 0.5, show.legend = FALSE, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f%%\n(n = %s)", pct, format(n, big.mark = ","))),
            vjust = -0.3, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = COLORES_CLASE) +
  scale_y_continuous(limits = c(0, 110), labels = function(x) paste0(x, "%")) +
  labs(title    = "Distribución de la Variable Objetivo",
       subtitle = "Desbalance de clases — principal desafío metodológico del PS",
       x = NULL, y = "Porcentaje de hogares",
       caption  = "Fuente: DANE — MESE 2018 | Bogotá") +
  tema_pub

guardar_fig(p1, "01_desbalance_clases", width = 7, height = 5)

# ------------------------------------------------------------------------------
# FIGURA 2 — Panel: tasa de pobreza por variables clave
# ------------------------------------------------------------------------------

cat("\n[Fig 2/6] Panel variables clave...\n")

labels_educ <- c("1" = "Ninguno", "2" = "Preescolar",
                 "3" = "Básica\nprimaria", "4" = "Básica\nsecundaria",
                 "5" = "Media", "6" = "Superior o\nuniversitaria")

var_educ  <- if ("jefe_educ"       %in% names(train)) "jefe_educ"       else
  if ("educ_jefe"       %in% names(train)) "educ_jefe"       else NULL
var_ocup  <- if ("prop_ocupados"   %in% names(train)) "prop_ocupados"   else
  if ("tasa_ocupacion"  %in% names(train)) "tasa_ocupacion"  else NULL
var_tam   <- if ("n_miembros"      %in% names(train)) "n_miembros"      else
  if ("n_personas"      %in% names(train)) "n_personas"      else
    if ("nper"            %in% names(train)) "nper"            else NULL
var_mujer <- if ("jefe_sexo_mujer" %in% names(train)) "jefe_sexo_mujer" else
  if ("jefe_mujer"      %in% names(train)) "jefe_mujer"      else NULL

p2a <- NULL
if (!is.null(var_educ)) {
  p2a <- train |>
    filter(!is.na(.data[[var_educ]])) |>
    group_by(educ = .data[[var_educ]]) |>
    summarise(tasa = mean(pobre == 1, na.rm = TRUE) * 100, n = n(), .groups = "drop") |>
    filter(n >= 10) |>
    mutate(educ_label = factor(labels_educ[as.character(educ)],
                               levels = unname(labels_educ))) |>
    ggplot(aes(x = educ_label, y = tasa)) +
    geom_col(fill = "#B71C1C", alpha = 0.85) +
    geom_text(aes(label = sprintf("%.0f%%", tasa)), vjust = -0.4, size = 3.5) +
    scale_y_continuous(limits = c(0, 105), labels = function(x) paste0(x, "%")) +
    labs(title = "Tasa de Pobreza por Nivel Educativo del Jefe de Hogar",
         subtitle = "A mayor educación del jefe, menor probabilidad de pobreza",
         x = "Nivel educativo del jefe", y = "% hogares pobres",
         caption = "Fuente: DANE — MESE 2018") +
    tema_pub
}

p2b <- NULL
if (!is.null(var_ocup)) {
  p2b <- train |>
    filter(!is.na(.data[[var_ocup]])) |>
    mutate(cat_ocup = cut(.data[[var_ocup]],
                          breaks = c(-Inf, 0, 0.25, 0.5, 0.75, Inf),
                          labels = c("0%", "1-25%", "26-50%", "51-75%", ">75%"))) |>
    group_by(cat_ocup) |>
    summarise(tasa = mean(pobre == 1, na.rm = TRUE) * 100, n = n(), .groups = "drop") |>
    filter(n >= 10) |>
    ggplot(aes(x = cat_ocup, y = tasa)) +
    geom_col(fill = "#E65100", alpha = 0.85) +
    geom_text(aes(label = sprintf("%.0f%%", tasa)), vjust = -0.4, size = 3.5) +
    scale_y_continuous(limits = c(0, 105), labels = function(x) paste0(x, "%")) +
    labs(title = "Tasa de Pobreza por Proporción de Ocupados",
         subtitle = "Más empleados en el hogar → menor pobreza",
         x = "Proporción de miembros empleados", y = "% hogares pobres",
         caption = "Fuente: DANE — MESE 2018") +
    tema_pub
}

p2c <- NULL
if (!is.null(var_tam)) {
  p2c <- train |>
    filter(!is.na(.data[[var_tam]])) |>
    mutate(tam_cat = pmin(.data[[var_tam]], 8)) |>
    group_by(tam_cat) |>
    summarise(tasa = mean(pobre == 1, na.rm = TRUE) * 100, n = n(), .groups = "drop") |>
    filter(n >= 10) |>
    ggplot(aes(x = factor(tam_cat), y = tasa)) +
    geom_col(fill = "#4527A0", alpha = 0.85) +
    geom_text(aes(label = sprintf("%.0f%%", tasa)), vjust = -0.4, size = 3.5) +
    scale_y_continuous(limits = c(0, 105), labels = function(x) paste0(x, "%")) +
    labs(title = "Tasa de Pobreza por Tamaño del Hogar",
         subtitle = "Hogares más grandes → mayor pobreza",
         x = "Número de miembros (8 = 8 o más)", y = "% hogares pobres",
         caption = "Fuente: DANE — MESE 2018") +
    tema_pub
}

p2d <- NULL
if (!is.null(var_mujer)) {
  p2d <- train |>
    filter(!is.na(.data[[var_mujer]])) |>
    mutate(genero = ifelse(.data[[var_mujer]] == 1, "Jefa mujer", "Jefe hombre")) |>
    group_by(genero) |>
    summarise(tasa = mean(pobre == 1, na.rm = TRUE) * 100, n = n(), .groups = "drop") |>
    ggplot(aes(x = genero, y = tasa, fill = genero)) +
    geom_col(width = 0.5, alpha = 0.9, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.1f%%\n(n=%s)", tasa, format(n, big.mark = ","))),
              vjust = -0.3, size = 4, fontface = "bold") +
    scale_fill_manual(values = c("Jefa mujer" = "#AD1457", "Jefe hombre" = "#1565C0")) +
    scale_y_continuous(limits = c(0, 105), labels = function(x) paste0(x, "%")) +
    labs(title = "Tasa de Pobreza por Sexo del Jefe de Hogar",
         subtitle = "Hogares con jefa mujer vs. jefe hombre",
         x = NULL, y = "% hogares pobres",
         caption = "Fuente: DANE — MESE 2018") +
    tema_pub
}

plots_panel <- Filter(Negate(is.null), list(p2a, p2b, p2c, p2d))
if (length(plots_panel) == 4) {
  guardar_fig((plots_panel[[1]] + plots_panel[[2]]) /
                (plots_panel[[3]] + plots_panel[[4]]),
              "02_panel_variables_clave", width = 14, height = 10)
} else if (length(plots_panel) >= 2) {
  guardar_fig(wrap_plots(plots_panel, ncol = 2),
              "02_panel_variables_clave", width = 14, height = 6)
}

# ------------------------------------------------------------------------------
# FIGURA 3 — Distribuciones por clase (violin + boxplot)
# ------------------------------------------------------------------------------

cat("\n[Fig 3/6] Distribuciones por clase...\n")

labels_vars <- c(
  "ratio_dependencia" = "Ratio de dependencia",
  "educ_max_hogar"    = "Educ. máxima del hogar",
  "educ_media_hogar"  = "Educ. media del hogar",
  "jefe_edad"         = "Edad del jefe",
  "edad_jefe"         = "Edad del jefe",
  "prop_ocupados"     = "Proporción de ocupados",
  "tasa_ocupacion"    = "Tasa de ocupación",
  "n_miembros"        = "N° personas del hogar",
  "nper"              = "N° personas del hogar",
  "hacinamiento"      = "Hacinamiento (pers/cuarto)",
  "prop_formal"       = "Proporción de formales"
)

vars_violin_candidatas <- c(
  "ratio_dependencia", "educ_max_hogar", "educ_media_hogar",
  "jefe_edad", "edad_jefe", "prop_ocupados", "tasa_ocupacion",
  "n_miembros", "nper", "hacinamiento", "prop_formal"
)
vars_violin <- intersect(vars_violin_candidatas, names(train))
vars_violin <- vars_violin[seq_len(min(6, length(vars_violin)))]

if (length(vars_violin) >= 2) {
  train_long <- train |>
    select(pobre, all_of(vars_violin)) |>
    mutate(clase = factor(pobre, labels = c("No pobre", "Pobre"))) |>
    select(-pobre) |>
    pivot_longer(cols = -clase, names_to = "variable", values_to = "valor") |>
    mutate(variable = recode(variable, !!!labels_vars))
  
  p3 <- ggplot(train_long, aes(x = clase, y = valor, fill = clase)) +
    geom_violin(alpha = 0.4, trim = TRUE, scale = "width") +
    geom_boxplot(width = 0.2, alpha = 0.8,
                 outlier.size = 0.5, outlier.alpha = 0.3) +
    scale_fill_manual(values = COLORES_CLASE) +
    facet_wrap(~ variable, scales = "free_y", ncol = 3) +
    labs(title    = "Distribución de Variables Clave por Condición de Pobreza",
         subtitle = "Comparación entre hogares pobres y no pobres — variables numéricas seleccionadas",
         x = NULL, y = "Valor", fill = "Condición",
         caption  = "Fuente: DANE — MESE 2018 | Bogotá") +
    tema_pub
  
  guardar_fig(p3, "03_distribuciones_por_clase", width = 14, height = 9)
}

# ------------------------------------------------------------------------------
# FIGURA 4 — Correlación con target (bar chart horizontal)
# ------------------------------------------------------------------------------

cat("\n[Fig 4/6] Correlación con target...\n")

top_corr <- correlaciones |>
  filter(!is.na(pobre)) |>
  arrange(desc(abs(pobre))) |>
  head(20) |>
  mutate(
    direccion = ifelse(pobre > 0, "Positiva", "Negativa"),
    variable  = fct_reorder(variable, pobre)
  )

p4 <- ggplot(top_corr, aes(x = variable, y = pobre, fill = direccion)) +
  geom_col(alpha = 0.85, show.legend = TRUE) +
  geom_text(aes(label = sprintf("%.3f", pobre),
                hjust = ifelse(pobre > 0, -0.1, 1.1)),
            size = 3.2) +
  coord_flip() +
  scale_fill_manual(values = c("Positiva" = "#B71C1C", "Negativa" = "#1565C0")) +
  scale_y_continuous(
    limits = c(min(top_corr$pobre) * 1.3, max(top_corr$pobre) * 1.3)
  ) +
  labs(title    = "Correlación de Variables con la Condición de Pobreza",
       subtitle = "Top 20 variables — correlación de Pearson con pobre (1 = pobre)",
       x = NULL, y = "Correlación con pobre", fill = "Dirección",
       caption  = "Fuente: DANE — MESE 2018 | Bogotá") +
  tema_pub

guardar_fig(p4, "04_correlacion_con_target", width = 11, height = 8)

# ------------------------------------------------------------------------------
# FIGURA 5 — Mapa de correlaciones entre variables
# ------------------------------------------------------------------------------

cat("\n[Fig 5/6] Mapa de correlaciones...\n")

vars_corr <- intersect(
  c("pobre", "n_miembros", "nper", "prop_ocupados", "tasa_ocupacion",
    "educ_max_hogar", "educ_media_hogar", "jefe_educ", "educ_jefe",
    "ratio_dependencia", "hacinamiento", "prop_formal",
    "jefe_sexo_mujer", "jefe_mujer", "jefe_trabaja",
    "n_menores_15", "n_mayores_65", "jefe_vulnerable"),
  names(train)
)

if (length(vars_corr) >= 3) {
  mat_corr <- train |>
    select(all_of(vars_corr)) |>
    mutate(across(everything(), as.numeric)) |>
    cor(use = "pairwise.complete.obs") |>
    as.data.frame() |>
    rownames_to_column("var1") |>
    pivot_longer(-var1, names_to = "var2", values_to = "corr")
  
  p5 <- ggplot(mat_corr, aes(x = var1, y = var2, fill = corr)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", corr)),
              size  = 2.8,
              color = ifelse(abs(mat_corr$corr) > 0.5, "white", "black")) +
    scale_fill_gradient2(
      low = "#1565C0", mid = "white", high = "#B71C1C",
      midpoint = 0, limits = c(-1, 1), name = "Correlación"
    ) +
    labs(title    = "Mapa de Correlaciones entre Variables Clave",
         subtitle = "Correlaciones de Pearson — variables seleccionadas del dataset",
         x = NULL, y = NULL,
         caption  = "Fuente: DANE — MESE 2018 | Bogotá") +
    tema_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
          axis.text.y = element_text(size = 9))
  
  guardar_fig(p5, "05_mapa_correlaciones", width = 12, height = 10)
}

# ------------------------------------------------------------------------------
# FIGURA 6 — Missings
# ------------------------------------------------------------------------------

cat("\n[Fig 6/6] Missings...\n")

if (nrow(missings_train) > 0) {
  p6 <- missings_train |>
    head(20) |>
    mutate(variable = fct_reorder(variable, pct_na)) |>
    ggplot(aes(x = variable, y = pct_na)) +
    geom_col(fill = "#F57F17", alpha = 0.85) +
    geom_text(aes(label = sprintf("%.1f%%", pct_na)), hjust = -0.1, size = 3.5) +
    coord_flip() +
    scale_y_continuous(limits = c(0, max(missings_train$pct_na) * 1.25),
                       labels = function(x) paste0(x, "%")) +
    labs(title    = "Variables con Valores Faltantes",
         subtitle = "Top 20 variables con mayor porcentaje de NAs en el train",
         x = NULL, y = "% de observaciones faltantes",
         caption  = "Fuente: DANE — MESE 2018 | Bogotá") +
    tema_pub
  
  guardar_fig(p6, "06_missings", width = 10, height = 7)
} else {
  cat("   Sin missings detectados en el dataset final ✓\n")
}

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

cat("\n>>> [EDA] Completado.\n")
cat(sprintf("   Figuras en: %s\n", here(paths$figures)))
cat(sprintf("   Tablas en:  %s\n", here(paths$tables)))
cat(sprintf("\n   Total hogares train:    %d\n", nrow(train)))
cat(sprintf("   Hogares pobres:         %d (%.1f%%)\n",
            sum(train$pobre == 1, na.rm = TRUE),
            mean(train$pobre == 1, na.rm = TRUE) * 100))
cat(sprintf("   Variables numéricas:    %d\n", length(vars_num)))
cat(sprintf("   Variables con missings: %d\n", nrow(missings_train)))