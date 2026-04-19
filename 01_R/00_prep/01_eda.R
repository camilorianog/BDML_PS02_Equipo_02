# ============================================================

# 01_eda.R

# Exploración descriptiva del dataset (post-exclusión de leakage)

# Ejecutar después de cargar paths desde 00_rundirectory.R

# ============================================================

suppressPackageStartupMessages({ library(tidyverse) library(skimr)
library(here) library(janitor) })

# --- Cargar datos limpios -----------------------------------

# Usar el cleaned output para EDA sobre las variables finales

train <- readRDS(here(paths$processed, "train_clean.rds"))
test  <- readRDS(here(paths$processed, "test_clean.rds"))

cat("Dims train:", dim(train), "\n") cat("Dims test: ", dim(test), "\n")

# ============================================================

# 1. TARGET

# ============================================================

cat("\n=== TARGET: Pobre ===\n") train \|\> count(pobre) \|\> mutate(pct
= round(n / sum(n) \* 100, 2)) \|\> print()

# Nuestro desbalance es de 20% pobres contra 80% no pobres

# ============================================================

# 2. VARIABLES FALTANTES

# ============================================================

cat("\n=== MISSING VALUES (train) ===\n") train \|\>
summarise(across(everything(), \~ sum(is.na(.)))) \|\>
pivot_longer(everything(), names_to = "variable", values_to = "n_na")
\|\> mutate(pct_na = round(n_na / nrow(train) \* 100, 2)) \|\>
filter(n_na \> 0) \|\> arrange(desc(pct_na)) \|\> print(n = 50)

# ============================================================

# 3. VARIABLES CATEGÓRICAS

# ============================================================

vars_cat \<- c("clase", "dominio", "depto", "p5090", "nivel_educ_max")

cat("\n=== DISTRIBUCIÓN VARIABLES CATEGÓRICAS ===\n") for (v in
vars_cat) { if (v %in% names(train)) { cat(sprintf("\n--- %s ---\n", v))
train \|\> count(.data[[v]], sort = FALSE) \|\> mutate(pct = round(n /
sum(n) \* 100, 1)) \|\> print(n = 30) } }

# ============================================================

# 4. VARIABLES NUMÉRICAS — ESTADÍSTICAS DESCRIPTIVAS

# ============================================================

cat("\n=== RESUMEN NUMÉRICO (skimr) ===\n") vars_num \<- train \|\>
select(-id, -pobre, -all_of(vars_cat[vars_cat %in% names(train)])) \|\>
select(where(is.numeric)) \|\> names()

train \|\> select(all_of(vars_num)) \|\> skim() \|\> print()

# ============================================================

# 5. DISTRIBUCIÓN POR CLASE (pobre vs no pobre)

# ============================================================

cat("\n=== MEDIAS POR GRUPO (pobre=0 vs pobre=1) ===\n") train \|\>
group_by(pobre) \|\> summarise(across(all_of(vars_num), \~
round(mean(.x, na.rm = TRUE), 3))) \|\> pivot_longer(-pobre, names_to =
"variable", values_to = "media") \|\> pivot_wider(names_from = pobre,
values_from = media, names_prefix = "pobre\_") \|\> mutate(dif_abs =
abs(pobre_1 - pobre_0), ratio = round(pobre_1 / pmax(pobre_0, 1e-6), 3))
\|\> arrange(desc(dif_abs)) \|\> print(n = 50)

# ============================================================

# 6. CORRELACIÓN CON TARGET

# ============================================================

cat("\n=== CORRELACIÓN CON POBRE (numérica, Pearson) ===\n") train \|\>
select(pobre, all_of(vars_num)) \|\> mutate(pobre = as.numeric(pobre))
\|\> cor(use = "pairwise.complete.obs") \|\> as.data.frame() \|\>
rownames_to_column("variable") \|\> select(variable, pobre) \|\>
filter(variable != "pobre") \|\> arrange(desc(abs(pobre))) \|\>
mutate(pobre = round(pobre, 4)) \|\> print(n = 50)

# ============================================================

# 7. OUTLIERS — PERCENTILES CLAVE

# ============================================================

cat("\n=== PERCENTILES 95-99-MAX (vars numéricas) ===\n") train \|\>
select(all_of(vars_num)) \|\> summarise(across(everything(), list( p95 =
\~ quantile(.x, 0.95, na.rm = TRUE), p99 = \~ quantile(.x, 0.99, na.rm =
TRUE), max = \~ max(.x, na.rm = TRUE) ))) \|\>
pivot_longer(everything(), names_to = c("variable", ".value"), names_sep
= "\_(?=[\^\_]+\$)") \|\> arrange(desc(max)) \|\> print(n = 60)

# ============================================================

# 8. CONSISTENCIA TRAIN vs TEST (columnas)

# ============================================================

cat("\n=== COLUMNAS EN TRAIN PERO NO EN TEST ===\n")
print(setdiff(names(train), names(test)))

cat("\n=== COLUMNAS EN TEST PERO NO EN TRAIN ===\n")
print(setdiff(names(test), names(train)))

# ============================================================

# 9. CORRELACIÓN NPER vs NPERSUG (redundancia)

# ============================================================

if (all(c("nper", "npersug") %in% names(train))) { cat(sprintf("\n===
CORR(nper, npersug) = %.4f ===\n", cor(train$nper, train$npersug, use =
"complete.obs"))) cat("Diferencia media:",
round(mean(train$nper - train$npersug, na.rm=TRUE), 4), "\n") cat("%
donde nper != npersug:", round(mean(train$nper != train$npersug,
na.rm=TRUE)\*100, 2), "%\n") }

# ============================================================

# 10. P5000 CON CÓDIGO 98/99 (raw data check)

# ============================================================

cat("\n=== P5000 VALORES EXTREMOS EN RAW ===\n") train_h_raw \<-
read.csv(here(paths$raw, "train_hogares.csv"))
cat("P5000 == 98:", sum(train_h_raw$P5000 == 98, na.rm=TRUE), "\n")
cat("P5000 == 99:", sum(train_h_raw$P5000 == 99, na.rm=TRUE), "\n")
cat("P5000 > 20:", sum(train_h_raw$P5000 \> 20, na.rm=TRUE), "\n")
rm(train_h_raw)

cat("\n\>\>\> 01_eda.R completado\n")
