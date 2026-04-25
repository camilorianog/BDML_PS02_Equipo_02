# ==============================================================================
# 02_eda_narrativa.R — Historia de selección de features con descriptivas + modelos
# ==============================================================================
# MECA 4107 · PS2 · Equipo 02
#
# Objetivo:
#   Construir una narrativa cuantitativa que justifique el feature set bajo
#   tres lentes que se refuerzan mutuamente:
#     (1) LITERATURA   — qué predice cada feature según la teoría
#     (2) DESCRIPTIVAS — diferencias pobre/no-pobre con effect sizes y tests
#     (3) MODELOS      — importancia y signo de coeficientes (Logit, Lasso, RF)
#
# Output:
#   Tablas (paths$tables)  — *.csv con los rankings y métricas
#   Figuras (paths$figures) — gráficos publication-quality 07_* a 11_*
#
# Asume cargado el entorno de 00_rundirectory.R (paths, SEED, CV_FOLDS, packages).
# Si se corre standalone, descomentar el bloque "STANDALONE BOOTSTRAP".
# ==============================================================================

# --- STANDALONE BOOTSTRAP (descomentar si se corre suelto) -------------------
# library(here)
# source(here::here("00_rundirectory.R"))   # solo cargará paquetes y paths

pacman::p_load(tidyverse, glmnet, ranger, broom, effectsize, rcompanion,
               patchwork, scales, gt)

cat("\n========================================================\n")
cat("  EDA NARRATIVO — Literatura × Descriptivas × Modelos\n")
cat("========================================================\n")

# --- Carga --------------------------------------------------------------------
train <- readRDS(here(paths$processed, "train_features.rds"))

# Casteos defensivos
train <- train |>
  mutate(
    pobre_bin = as.integer(as.character(pobre)),
    pobre_fac = factor(pobre_bin, levels = c(0, 1),
                       labels = c("no_pobre", "pobre"))
  )

# ==============================================================================
# BLOQUE 1 — CATÁLOGO DE FEATURES CON HIPÓTESIS DE LITERATURA
# ==============================================================================
# Cada feature engineered se anota con:
#   - hipótesis (signo esperado del efecto sobre P(pobre))
#   - cita corta
#   - mecanismo económico
# Esto permite testear si la dirección observada coincide con la teoría.

catalogo <- tribble(
  ~feature,                ~signo_esperado, ~cita,                                  ~mecanismo,
  "ratio_dependencia",     "+",  "IDB Costa Rica 2018",                  "Más dependientes por ocupado → menor ingreso per cápita",
  "hacinamiento",          "+",  "Banerjee 2018; UN-Habitat 2020",       "Densidad habitacional alta es proxy de calidad de vida baja",
  "educ_x_ocup",           "-",  "Nkurunziza 2024; Marrugo-Arnedo 2015", "Capital humano solo paga si está empleado",
  "rural_x_ocup",          "+",  "Obando-Andrián 2015; WB 2019",         "Empleo rural = baja productividad, alta informalidad",
  "formal_x_salud",        "-",  "WB 2019; UNDP-ECLAC 2024",             "Formalidad plena = ingreso estable y protegido",
  "nper_sq",               "+",  "UN Statistics 2005; SOAS 2005",        "Hogares grandes — efecto no lineal sobre pobreza",
  "edad_prom_sq",          "+",  "Obando-Andrián 2015; Banerjee 2018",   "U-invertida: muy jóvenes y muy viejos más vulnerables",
  "mujeres_x_inact",       "+",  "Corral 2024; UNDP-ECLAC 2024",         "Brecha de género laboral × falta de ingreso",
  "jefe_mujer_inact",      "+",  "Bleynat 2020; Chant 2003",             "Jefatura femenina pesa cuando se combina con inactividad",
  "tasa_inactivos",        "+",  "DANE GEIH 2018",                       "Fracción de PET sin trabajo → menor ingreso",
  "sin_ocupados",          "+",  "WB 2019",                              "Hogar sin ocupados = vulnerabilidad extrema",
  "educ_jefe_x_ocup",      "-",  "Nkurunziza 2024",                      "Capital humano del jefe activado por empleo",
  "calidad_empleo",        "-",  "WB 2019; UNDP-ECLAC 2024",             "Horas × formalidad como proxy de ingreso laboral",
  "presion_habitacional",  "+",  "UN-Habitat 2020",                      "Cuartos usados / disponibles — densidad efectiva",
  "jefe_vulnerable",       "+",  "Corral 2024; Nkurunziza 2024",         "Jefe sin empleo y baja educación → pobreza crónica",
  "doble_proteccion",      "-",  "UNDP-ECLAC 2024",                      "Cobertura laboral completa → ingreso estable",
  "ratio_mayores_65",      "+",  "WB 2019",                              "Carga de vejez sobre PET",
  "jefe_mayor_inactivo",   "+",  "Banerjee 2018",                        "Jefe >60 sin empleo → dependencia de transferencias",
  # Variables base clave (no engineered pero centrales)
  "tasa_ocupacion",        "-",  "DANE GEIH",                            "Más ocupados en el hogar → más ingreso",
  "nper",                  "+",  "UN Statistics 2005",                   "Más bocas que alimentar",
  "edad_jefe",             "?",  "Banerjee 2018",                        "Relación U con pobreza",
  "jefe_mujer",            "+",  "Chant 2003",                           "Asociado a feminización de la pobreza",
  "prop_cotiza_pension",   "-",  "WB 2019",                              "Formalidad — ingreso protegido",
  "prop_afiliado_salud",   "-",  "UNDP-ECLAC 2024",                      "Acceso a salud contributiva = formalidad",
  "horas_trabajo_prom",    "-",  "DANE GEIH",                            "Horas trabajadas → ingreso laboral"
)

write_csv(catalogo, here(paths$tables, "07_catalogo_features.csv"))
cat(sprintf("  [1/5] Catálogo guardado: %d features anotadas\n", nrow(catalogo)))

# ==============================================================================
# BLOQUE 2 — DESCRIPTIVAS POR GRUPO + EFFECT SIZES + TESTS
# ==============================================================================

vars_test <- intersect(catalogo$feature, names(train))
vars_test <- vars_test[sapply(train[vars_test], is.numeric)]

# Helper: Cohen's d, t-test, correlación punto-biserial
descriptiva_var <- function(v) {
  x0 <- train[[v]][train$pobre_bin == 0]
  x1 <- train[[v]][train$pobre_bin == 1]
  d  <- tryCatch(effectsize::cohens_d(train[[v]], train$pobre_fac)$Cohens_d,
                 error = function(e) NA_real_)
  tt <- tryCatch(t.test(x1, x0, var.equal = FALSE), error = function(e) NULL)
  cr <- tryCatch(cor(train[[v]], train$pobre_bin, use = "pairwise.complete.obs"),
                 error = function(e) NA_real_)
  tibble(
    feature      = v,
    media_no_pob = mean(x0, na.rm = TRUE),
    media_pob    = mean(x1, na.rm = TRUE),
    sd_no_pob    = sd(x0, na.rm = TRUE),
    sd_pob       = sd(x1, na.rm = TRUE),
    diff_medias  = mean(x1, na.rm = TRUE) - mean(x0, na.rm = TRUE),
    cohens_d     = -d,                          # signo positivo = mayor en pobres
    corr_pearson = cr,
    t_stat       = if (!is.null(tt)) unname(tt$statistic) else NA_real_,
    p_value      = if (!is.null(tt)) tt$p.value           else NA_real_
  )
}

cat("  [2/5] Calculando descriptivas + effect sizes...\n")
desc <- map_dfr(vars_test, descriptiva_var) |>
  left_join(catalogo, by = "feature") |>
  mutate(
    direccion_obs = case_when(diff_medias > 0 ~ "+", diff_medias < 0 ~ "-",
                              TRUE ~ "0"),
    signo_ok      = direccion_obs == signo_esperado |
                    signo_esperado == "?",
    magnitud      = case_when(
      abs(cohens_d) >= 0.8 ~ "grande",
      abs(cohens_d) >= 0.5 ~ "moderado",
      abs(cohens_d) >= 0.2 ~ "pequeño",
      TRUE                 ~ "trivial"
    )
  ) |>
  arrange(desc(abs(cohens_d)))

write_csv(desc, here(paths$tables, "08_descriptivas_effect_sizes.csv"))
cat(sprintf("       %d features procesadas | %d con signo esperado correcto\n",
            nrow(desc), sum(desc$signo_ok, na.rm = TRUE)))

# ==============================================================================
# BLOQUE 3 — IMPORTANCIA EN MODELOS (Logit, Lasso, RF)
# ==============================================================================
# Re-fit ligero para extraer:
#   - Logit: coeficientes z-estandarizados (signo + magnitud)
#   - Lasso: coeficientes con λ.1se (selección parsimoniosa)
#   - RF:    permutation importance
# ==============================================================================

# Matriz de diseño común — solo features del catálogo presentes
feats_modelo <- intersect(catalogo$feature, names(train))
feats_modelo <- feats_modelo[sapply(train[feats_modelo], is.numeric)]

X_df <- train |> select(all_of(feats_modelo))
y    <- train$pobre_bin

# Estandarización para coeficientes comparables (solo para logit/lasso)
X_std <- scale(as.matrix(X_df))
X_std[is.na(X_std)] <- 0

# --- 3a. Logit z-estandarizado -----------------------------------------------
cat("  [3/5] Logit (coefs z-estandarizados)...\n")
fit_logit <- glm(y ~ X_std, family = binomial())
coef_logit <- broom::tidy(fit_logit) |>
  filter(term != "(Intercept)") |>
  mutate(feature = sub("^X_std", "", term)) |>
  select(feature, estimate_logit = estimate, p_logit = p.value)

# --- 3b. Lasso (cv.glmnet, λ.1se) --------------------------------------------
cat("       Lasso (cv.glmnet, λ.1se)...\n")
set.seed(SEED)
cvl <- cv.glmnet(X_std, y, family = "binomial", alpha = 1, nfolds = 5)
coef_lasso <- as.matrix(coef(cvl, s = "lambda.1se"))
coef_lasso <- tibble(
  feature        = rownames(coef_lasso),
  estimate_lasso = as.numeric(coef_lasso)
) |> filter(feature != "(Intercept)") |>
  mutate(feature = sub("^X_std", "", feature))

# --- 3c. Random Forest permutation importance --------------------------------
cat("       Random Forest (ntree=100, permutation importance)...\n")
set.seed(SEED)
rf <- ranger(
  y = factor(y), x = X_df,
  num.trees    = 100,
  importance   = "permutation",
  num.threads  = max(1, parallel::detectCores() - 1),
  probability  = FALSE
)
imp_rf <- tibble(
  feature   = names(rf$variable.importance),
  imp_rf    = as.numeric(rf$variable.importance)
)

# --- 3d. Consolidación + ranking de consenso --------------------------------
ranking <- desc |>
  select(feature, signo_esperado, cita, mecanismo,
         cohens_d, corr_pearson, p_value) |>
  left_join(coef_logit, by = "feature") |>
  left_join(coef_lasso, by = "feature") |>
  left_join(imp_rf,     by = "feature") |>
  mutate(
    rank_d     = rank(-abs(cohens_d),       na.last = "keep"),
    rank_logit = rank(-abs(estimate_logit), na.last = "keep"),
    rank_lasso = rank(-abs(estimate_lasso), na.last = "keep"),
    rank_rf    = rank(-imp_rf,              na.last = "keep"),
    rank_avg   = rowMeans(across(c(rank_d, rank_logit, rank_lasso, rank_rf)),
                          na.rm = TRUE)
  ) |>
  arrange(rank_avg)

write_csv(ranking, here(paths$tables, "09_ranking_consenso.csv"))
cat(sprintf("       Tabla consolidada: %d features × %d métricas\n",
            nrow(ranking), ncol(ranking)))

# ==============================================================================
# BLOQUE 4 — VISUALIZACIONES NARRATIVAS
# ==============================================================================

guardar_fig <- function(p, nombre, w = 12, h = 7, dpi = 300) {
  ggsave(here(paths$figures, paste0(nombre, ".png")),
         plot = p, width = w, height = h, dpi = dpi, bg = "white")
  cat(sprintf("       Figura: %s.png\n", nombre))
}

tema_pub <- theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", size = 14, color = "#0A2240"),
        plot.subtitle = element_text(size = 11, color = "#555555"),
        plot.caption  = element_text(size = 9,  color = "#888888", hjust = 0),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "#EEEEEE"),
        legend.position = "bottom")

cat("  [4/5] Generando figuras...\n")

# --- Fig 7. Forest de Cohen's d coloreado por hipótesis ----------------------
fig7 <- desc |>
  filter(!is.na(cohens_d)) |>
  mutate(
    feature = fct_reorder(feature, cohens_d),
    coincide = ifelse(signo_ok, "Coincide con literatura", "Contradice literatura")
  ) |>
  ggplot(aes(x = cohens_d, y = feature, color = coincide)) +
  geom_vline(xintercept = 0, color = "grey40", linetype = "dashed") +
  geom_vline(xintercept = c(-0.2, 0.2), color = "grey80", linetype = "dotted") +
  geom_vline(xintercept = c(-0.5, 0.5), color = "grey80", linetype = "dotted") +
  geom_segment(aes(x = 0, xend = cohens_d, yend = feature), linewidth = 0.6) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Coincide con literatura"  = "#1565C0",
                                "Contradice literatura"    = "#B71C1C")) +
  labs(title    = "Effect Size (Cohen's d) por Feature",
       subtitle = "Magnitud y dirección de la diferencia entre hogares pobres y no pobres",
       x = "Cohen's d  (positivo = mayor en pobres)", y = NULL,
       color = NULL,
       caption = "Líneas punteadas: |d| = 0.2 (pequeño), 0.5 (moderado). Fuente: DANE MESE 2018") +
  tema_pub

guardar_fig(fig7, "07_cohens_d_forest", w = 11, h = 9)

# --- Fig 8. Ranking cross-modelo (heatmap de ranks) --------------------------
fig8_dat <- ranking |>
  slice_head(n = 20) |>
  select(feature, rank_d, rank_logit, rank_lasso, rank_rf) |>
  pivot_longer(-feature, names_to = "fuente", values_to = "rango") |>
  mutate(
    fuente  = recode(fuente,
                     rank_d     = "Cohen's d",
                     rank_logit = "Logit |z|",
                     rank_lasso = "Lasso |β|",
                     rank_rf    = "RF importance"),
    feature = fct_rev(fct_inorder(feature))
  )

fig8 <- ggplot(fig8_dat, aes(x = fuente, y = feature, fill = rango)) +
  geom_tile(color = "white") +
  geom_text(aes(label = ifelse(is.na(rango), "—", as.integer(rango))),
            size = 3.2, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "#0A2240", high = "#B0BEC5",
                      name = "Rango (1 = más importante)",
                      na.value = "grey90") +
  labs(title    = "Importancia Cross-Modelo — Top 20 Features",
       subtitle = "Las features que aparecen en color oscuro en TODAS las columnas son las más robustas",
       x = NULL, y = NULL,
       caption = "Rangos calculados sobre |Cohen's d|, |coef Logit z|, |coef Lasso|, importancia permutación RF") +
  tema_pub +
  theme(panel.grid = element_blank(),
        axis.text.y = element_text(size = 10))

guardar_fig(fig8, "08_ranking_cross_modelo", w = 10, h = 9)

# --- Fig 9. Lollipop signed coefficients del Logit --------------------------
fig9 <- coef_logit |>
  left_join(catalogo |> select(feature, signo_esperado), by = "feature") |>
  arrange(desc(abs(estimate_logit))) |>
  slice_head(n = 20) |>
  mutate(
    feature = fct_reorder(feature, estimate_logit),
    sig     = ifelse(p_logit < 0.01, "p<0.01", "p≥0.01")
  ) |>
  ggplot(aes(x = estimate_logit, y = feature)) +
  geom_vline(xintercept = 0, color = "grey40", linetype = "dashed") +
  geom_segment(aes(x = 0, xend = estimate_logit, yend = feature),
               color = "grey50") +
  geom_point(aes(color = sig), size = 4) +
  scale_color_manual(values = c("p<0.01" = "#0A2240", "p≥0.01" = "grey60")) +
  labs(title    = "Coeficientes Logit (estandarizados) — Top 20 |β|",
       subtitle = "Signo y magnitud sobre log-odds de pobreza, controlando por las demás features",
       x = "Coeficiente estandarizado", y = NULL, color = "Significancia",
       caption = "Variables estandarizadas (z) — coeficientes directamente comparables") +
  tema_pub

guardar_fig(fig9, "09_logit_signed_coefs", w = 10, h = 8)

# --- Fig 10. Storyboard top-6 features: distribución pobre vs no pobre ------
top6 <- ranking |> slice_head(n = 6) |> pull(feature)

storyboard_dat <- train |>
  select(pobre_fac, all_of(top6)) |>
  pivot_longer(-pobre_fac, names_to = "feature", values_to = "valor") |>
  left_join(catalogo |> select(feature, cita), by = "feature")

fig10 <- ggplot(storyboard_dat,
                aes(x = valor, fill = pobre_fac, color = pobre_fac)) +
  geom_density(alpha = 0.35, linewidth = 0.6) +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  scale_fill_manual(values  = c("no_pobre" = "#1565C0", "pobre" = "#B71C1C"),
                    labels = c("No pobre", "Pobre")) +
  scale_color_manual(values = c("no_pobre" = "#1565C0", "pobre" = "#B71C1C"),
                     labels = c("No pobre", "Pobre")) +
  labs(title    = "Top 6 Features por Consenso Cross-Modelo",
       subtitle = "Distribución de pobres vs no pobres — separación visual valida la importancia estadística",
       x = NULL, y = "Densidad", fill = NULL, color = NULL,
       caption = "Ranking promedio de Cohen's d, Logit, Lasso y RF") +
  tema_pub

guardar_fig(fig10, "10_storyboard_top_features", w = 14, h = 8)

# --- Fig 11. Coincidencia signo esperado vs observado -----------------------
fig11_dat <- desc |>
  filter(signo_esperado != "?") |>
  count(signo_ok) |>
  mutate(pct = n / sum(n) * 100,
         lab = ifelse(signo_ok, "Coincide", "Contradice"))

fig11 <- ggplot(fig11_dat, aes(x = lab, y = pct, fill = lab)) +
  geom_col(width = 0.5, alpha = 0.9, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.0f%%\n(n=%d)", pct, n)),
            vjust = -0.3, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Coincide" = "#1565C0", "Contradice" = "#B71C1C")) +
  scale_y_continuous(limits = c(0, 110), labels = function(x) paste0(x, "%")) +
  labs(title    = "Validación de Hipótesis de Literatura",
       subtitle = "Porcentaje de features con signo observado igual al predicho por la teoría",
       x = NULL, y = "% de features",
       caption = "Excluye features con signo esperado ambiguo") +
  tema_pub

guardar_fig(fig11, "11_validacion_literatura", w = 7, h = 5)

# ==============================================================================
# BLOQUE 5 — RESUMEN EJECUTIVO
# ==============================================================================

cat("\n  [5/5] Resumen ejecutivo:\n")
cat(sprintf("     · Features evaluadas:                 %d\n", nrow(desc)))
cat(sprintf("     · Con signo esperado correcto:        %d (%.0f%%)\n",
            sum(desc$signo_ok, na.rm = TRUE),
            mean(desc$signo_ok, na.rm = TRUE) * 100))
cat(sprintf("     · Con effect size moderado o grande:  %d\n",
            sum(abs(desc$cohens_d) >= 0.5, na.rm = TRUE)))
cat(sprintf("     · Seleccionadas por Lasso (β≠0):      %d\n",
            sum(coef_lasso$estimate_lasso != 0)))

cat("\n     Top 10 features por consenso cross-modelo:\n")
print(ranking |> slice_head(n = 10) |>
        select(feature, cohens_d, estimate_logit, imp_rf, rank_avg) |>
        mutate(across(where(is.numeric), ~ round(., 3))))

cat("\n>>> EDA narrativo completado.\n")
cat(sprintf("    Tablas:  %s\n", here(paths$tables)))
cat(sprintf("    Figuras: %s\n", here(paths$figures)))
