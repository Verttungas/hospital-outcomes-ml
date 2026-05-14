#!/usr/bin/env Rscript
# =============================================================================
# Trabajo Final — Machine Learning
# Equipo: Nicolás Barrón Sour · Ximena Flores Martínez · Carlos Vertti · Paloma Gutiérrez
# Universidad Anáhuac México — Dra. María del Carmen Villar Patiño
# -----------------------------------------------------------------------------
# Script de despliegue del mejor modelo: Red Neuronal (nnet) con
# size = 10, decay = 0.1, maxit = 120 — la mejor combinación encontrada en
# CV sobre el reporte (AUC en CP ≈ 0.949 vs Árbol 0.933 y SVM 0.925).
#
# Hace lo que pide el inciso 5 de la rúbrica:
#   a) Lee data/pruebaTF.csv
#   b) Aplica las transformaciones del pipeline (idénticas a las de la Parte 2 del reporte)
#   c) Divide en CE y CP (60 / 40) con la semilla 2711
#   d) Entrena el modelo con los hiperparámetros establecidos
#   e) Aplica el algoritmo al CP
#   f) Reporta Accuracy (caret::confusionMatrix) y ROC-AUC (pROC::roc)
#
# Uso:
#   Rscript scripts/predecir_TF.R
#
# Dependencias: caret, e1071, nnet, pROC, dplyr
# =============================================================================

suppressPackageStartupMessages({
  library(caret)
  library(e1071)
  library(nnet)    # Red Neuronal — el modelo final
  library(pROC)
  library(dplyr)   # cargar al final para que dplyr::select gane sobre MASS::select
})

# -----------------------------------------------------------------------------
# 1. Carga del archivo de prueba
# -----------------------------------------------------------------------------
ruta_csv <- "data/pruebaTF.csv"
if (!file.exists(ruta_csv)) {
  stop("No se encontró '", ruta_csv,
       "'. Coloca el archivo en data/ antes de ejecutar el script.")
}

df <- read.csv(ruta_csv, stringsAsFactors = FALSE, na.strings = c("", "NA"))

# -----------------------------------------------------------------------------
# 2. Helper: mapeo de CIE-10 a sus 22 capítulos
# -----------------------------------------------------------------------------
mapear_cie10 <- function(codigo) {
  codigo_limpio <- toupper(trimws(as.character(codigo)))
  primera       <- substr(codigo_limpio, 1, 1)
  dos_digitos   <- suppressWarnings(as.integer(substr(codigo_limpio, 2, 3)))

  dplyr::case_when(
    primera %in% c("A", "B")                                                  ~ "I_Infecciosas",
    primera == "C"                                                             ~ "II_Neoplasias",
    primera == "D" & !is.na(dos_digitos) & dos_digitos <= 49                   ~ "II_Neoplasias",
    primera == "D" & !is.na(dos_digitos) & dos_digitos >= 50 & dos_digitos <= 89 ~ "III_SangreInmunidad",
    primera == "E"                                                             ~ "IV_Endocrinas",
    primera == "F"                                                             ~ "V_Mental",
    primera == "G"                                                             ~ "VI_Nervioso",
    primera == "H" & !is.na(dos_digitos) & dos_digitos <= 59                   ~ "VII_Ojo",
    primera == "H" & !is.na(dos_digitos) & dos_digitos >= 60 & dos_digitos <= 95 ~ "VIII_Oido",
    primera == "I"                                                             ~ "IX_Circulatorio",
    primera == "J"                                                             ~ "X_Respiratorio",
    primera == "K"                                                             ~ "XI_Digestivo",
    primera == "L"                                                             ~ "XII_Piel",
    primera == "M"                                                             ~ "XIII_Musculoesqueletico",
    primera == "N"                                                             ~ "XIV_Genitourinario",
    primera == "O"                                                             ~ "XV_Embarazo",
    primera == "P"                                                             ~ "XVI_Perinatal",
    primera == "Q"                                                             ~ "XVII_Congenitas",
    primera == "R"                                                             ~ "XVIII_SintomasSignos",
    primera %in% c("S", "T")                                                   ~ "XIX_Traumatismos",
    primera %in% c("V", "W", "X", "Y")                                         ~ "XX_CausasExternas",
    primera == "Z"                                                             ~ "XXI_FactoresSalud",
    primera == "U"                                                             ~ "XXII_Especiales",
    TRUE                                                                       ~ "OTRO"
  )
}

# -----------------------------------------------------------------------------
# 3. Transformaciones (idénticas a las de Parte 2 del reporte)
# -----------------------------------------------------------------------------
df <- df |>
  mutate(
    EGRESO    = factor(EGRESO, levels = c("Defuncion", "Mejoria")),
    CIE10_CAP = mapear_cie10(CIE10)
  )

# 3.1 Eliminar categorías minoritarias en GENERO y ESTADO
df <- df |>
  filter(GENERO != "N.E.") |>
  group_by(ESTADO) |> filter(n() >= 10) |> ungroup()

# 3.2 Anomalía de captura en Sinaloa (PESO = 99 por defecto)
df <- df |>
  mutate(PESO = if_else(ESTADO == "Sinaloa" & PESO == 99, NA_real_, PESO))

# 3.3 Recorte de outliers PESO / ALTURA por percentiles 1 y 99
p1_peso  <- quantile(df$PESO,   0.01, na.rm = TRUE)
p99_peso <- quantile(df$PESO,   0.99, na.rm = TRUE)
p1_alt   <- quantile(df$ALTURA, 0.01, na.rm = TRUE)
p99_alt  <- quantile(df$ALTURA, 0.99, na.rm = TRUE)

df <- df |>
  filter(is.na(PESO)   | between(PESO,   p1_peso, p99_peso)) |>
  filter(is.na(ALTURA) | between(ALTURA, p1_alt,  p99_alt))

# 3.4 EDAD > 120 y DIAS_ESTANCIA > P99 → NA
df <- df |>
  mutate(EDAD = if_else(EDAD > 120, NA_real_, EDAD))

p99_dias <- quantile(df$DIAS_ESTANCIA, 0.99, na.rm = TRUE)
df <- df |>
  mutate(DIAS_ESTANCIA = if_else(DIAS_ESTANCIA > p99_dias, NA_real_, DIAS_ESTANCIA))

# 3.5 Imputación PESO / ALTURA por mediana de (CIE10_CAP × GENERO)
df <- df |>
  group_by(CIE10_CAP, GENERO) |>
  mutate(
    PESO   = if_else(is.na(PESO),   median(PESO,   na.rm = TRUE), PESO),
    ALTURA = if_else(is.na(ALTURA), median(ALTURA, na.rm = TRUE), ALTURA)
  ) |> ungroup()

# Respaldo: mediana global para grupos sin suficientes datos
med_peso   <- median(df$PESO,   na.rm = TRUE)
med_altura <- median(df$ALTURA, na.rm = TRUE)
df <- df |>
  mutate(
    PESO   = if_else(is.na(PESO),   med_peso,   PESO),
    ALTURA = if_else(is.na(ALTURA), med_altura, ALTURA)
  )

# 3.6 ESTADO → REGION (4 regiones SSA)
df <- df |>
  mutate(REGION = case_when(
    ESTADO %in% c("Baja California", "Baja California Sur", "Chihuahua",
                  "Coahuila de Zaragoza", "Durango", "Nuevo Leon",
                  "Sinaloa", "Sonora", "Tamaulipas")                          ~ "REGION_I",
    ESTADO %in% c("Aguascalientes", "Colima", "Guanajuato", "Jalisco",
                  "Michoacan de Ocampo", "Nayarit", "Queretaro de Arteaga",
                  "San Luis Potosi", "Zacatecas")                             ~ "REGION_II",
    ESTADO %in% c("Distrito Federal", "Mexico", "Guerrero",
                  "Hidalgo", "Morelos", "Puebla", "Tlaxcala")                 ~ "REGION_III",
    ESTADO %in% c("Campeche", "Chiapas", "Oaxaca", "Quintana Roo",
                  "Tabasco", "Veracruz de Ignacio de la Llave", "Yucatan")    ~ "REGION_IV",
    ESTADO == "No Especificado"                                               ~ "NO_ESPECIFICADO",
    TRUE                                                                       ~ "OTRA"
  )) |>
  select(-ESTADO)

# 3.7 ASEGURADO → ASEGURADO_GRP (6 grupos)
df <- df |>
  mutate(ASEGURADO_GRP = case_when(
    ASEGURADO == "IMSS"                                       ~ "IMSS",
    ASEGURADO == "ISSSTE"                                     ~ "ISSSTE",
    ASEGURADO %in% c("SEGURO POPULAR", "SPSS")               ~ "SEGURO_POPULAR",
    ASEGURADO %in% c("SIN SEGURO", "NINGUNA", "NO ASEGURADO") ~ "SIN_SEGURO",
    ASEGURADO == "SE IGNORA"                                  ~ "SE_IGNORA",
    TRUE                                                       ~ "OTRO"
  )) |>
  select(-ASEGURADO)

# 3.8 ACCIDENTE → binaria
df <- df |>
  mutate(ACCIDENTE = if_else(ACCIDENTE == "NO APLICA", "NO", "SI"))

# 3.9 Codificación de factores
df <- df |>
  mutate(
    EGRESO        = factor(EGRESO, levels = c("Defuncion", "Mejoria")),
    GENERO        = as.factor(GENERO),
    INFECCION     = as.factor(INFECCION),
    ACCIDENTE     = as.factor(ACCIDENTE),
    INDIGENA      = as.factor(INDIGENA),
    CIE10_CAP     = as.factor(CIE10_CAP),
    REGION        = as.factor(REGION),
    ASEGURADO_GRP = as.factor(ASEGURADO_GRP),
    MES_INGRESO   = factor(MES_INGRESO,
                           levels = c("Enero","Febrero","Marzo","Abril","Mayo","Junio",
                                      "Julio","Agosto","Septiembre","Octubre",
                                      "Noviembre","Diciembre"),
                           ordered = FALSE)
  ) |>
  select(-CIE10)

cat(sprintf("Datos preparados: %d filas × %d columnas\n", nrow(df), ncol(df)))

# -----------------------------------------------------------------------------
# 4. División 60 / 40 con la semilla 2711
# -----------------------------------------------------------------------------
set.seed(2711)
idx_train <- createDataPartition(df$EGRESO, p = 0.60, list = FALSE)
df_train  <- df[ idx_train, ]
df_test   <- df[-idx_train, ]

cat(sprintf("CE: %d filas (%.1f%% Defunción)\n",
            nrow(df_train), 100 * mean(df_train$EGRESO == "Defuncion")))
cat(sprintf("CP: %d filas (%.1f%% Defunción)\n",
            nrow(df_test), 100 * mean(df_test$EGRESO == "Defuncion")))

# ---- Imputación de NAs residuales (EDAD y DIAS_ESTANCIA) ----
# Se computa la mediana sobre CE únicamente y se aplica a ambos conjuntos.
# Evita data leakage y evita que predict() descarte filas con NA.
med_edad <- median(df_train$EDAD,          na.rm = TRUE)
med_dias <- median(df_train$DIAS_ESTANCIA, na.rm = TRUE)

df_train$EDAD          <- ifelse(is.na(df_train$EDAD),          med_edad, df_train$EDAD)
df_train$DIAS_ESTANCIA <- ifelse(is.na(df_train$DIAS_ESTANCIA), med_dias, df_train$DIAS_ESTANCIA)
df_test$EDAD           <- ifelse(is.na(df_test$EDAD),           med_edad, df_test$EDAD)
df_test$DIAS_ESTANCIA  <- ifelse(is.na(df_test$DIAS_ESTANCIA),  med_dias, df_test$DIAS_ESTANCIA)

# -----------------------------------------------------------------------------
# 5. Entrenamiento del modelo final
# -----------------------------------------------------------------------------
# Hiperparámetros establecidos en Parte 3 del reporte (mejor combinación de
# la rejilla con 5-fold CV + upsampling intra-pliegue):
#   método: Red Neuronal (nnet)
#   size  = 10  (neuronas en la capa oculta)
#   decay = 0.1 (regularización L2)
#   maxit = 120 (iteraciones de optimización)
#   preProcess: range (normalización a [0,1])
# Estrategia de balanceo: upsampling de la clase positiva (Defunción) sobre CE.

set.seed(2711)
df_train_bal <- upSample(
  x     = df_train[, setdiff(names(df_train), "EGRESO")],
  y     = df_train$EGRESO,
  yname = "EGRESO"
)

ctrl_final <- trainControl(method = "none", classProbs = TRUE)

set.seed(2711)
modelo_final <- train(
  EGRESO ~ .,
  data       = df_train_bal,
  method     = "nnet",
  preProcess = c("range"),
  tuneGrid   = data.frame(size = 10, decay = 0.1),
  trControl  = ctrl_final,
  MaxNWts    = 2000,
  maxit      = 120,
  trace      = FALSE
)

# -----------------------------------------------------------------------------
# 6. Aplicar al CP
# -----------------------------------------------------------------------------
prob_cp <- predict(modelo_final, df_test, type = "prob")[, "Defuncion"]
pred_cp <- predict(modelo_final, df_test)

# -----------------------------------------------------------------------------
# 7. Métricas de desempeño
# -----------------------------------------------------------------------------
cat("\n========== DESEMPEÑO SOBRE CP ==========\n")

cm <- confusionMatrix(pred_cp, df_test$EGRESO,
                      positive = "Defuncion", mode = "everything")
print(cm)

roc_obj <- pROC::roc(df_test$EGRESO, prob_cp,
                     levels = c("Mejoria", "Defuncion"),
                     direction = "<", quiet = TRUE)

cat(sprintf("\nAccuracy en CP: %.4f\n", as.numeric(cm$overall["Accuracy"])))
cat(sprintf("ROC-AUC  en CP: %.4f\n", as.numeric(pROC::auc(roc_obj))))
