#!/usr/bin/env Rscript
# =============================================================================
# Trabajo Final — Machine Learning
# Pre-entrenamiento de los modelos para reporte.qmd
# -----------------------------------------------------------------------------
# Replica el pipeline de Parte 1+2 del reporte, entrena los 6 modelos base
# + los 2 modelos refinados de Parte 3.7, y guarda todo en models/*.rds.
#
# Una vez ejecutado, reporte.qmd renderiza en pocos minutos cargando los
# modelos pre-entrenados.
#
# Uso:
#   Rscript scripts/entrenar_modelos.R
# =============================================================================

suppressPackageStartupMessages({
  library(caret)
  library(e1071)
  library(klaR)
  library(kernlab)
  library(rpart)
  library(nnet)
  library(doParallel)
  library(dplyr)  # cargar al final para que dplyr::select gane sobre MASS::select
})

SEED        <- 2711
RUTA_CSV    <- "data/TF_defunciones.csv"
DIR_MODELOS <- "models"
N_CORES     <- max(1, parallel::detectCores() - 2)

if (!dir.exists(DIR_MODELOS)) dir.create(DIR_MODELOS, recursive = TRUE)
if (!file.exists(RUTA_CSV)) stop("No se encontró ", RUTA_CSV)

# ---- helper: CIE10 -> capítulos ----
mapear_cie10 <- function(codigo) {
  cl <- toupper(trimws(as.character(codigo)))
  p1 <- substr(cl, 1, 1)
  d2 <- suppressWarnings(as.integer(substr(cl, 2, 3)))
  dplyr::case_when(
    p1 %in% c("A","B")                                  ~ "I_Infecciosas",
    p1 == "C"                                            ~ "II_Neoplasias",
    p1 == "D" & !is.na(d2) & d2 <= 49                    ~ "II_Neoplasias",
    p1 == "D" & !is.na(d2) & d2 >= 50 & d2 <= 89         ~ "III_SangreInmunidad",
    p1 == "E"                                            ~ "IV_Endocrinas",
    p1 == "F"                                            ~ "V_Mental",
    p1 == "G"                                            ~ "VI_Nervioso",
    p1 == "H" & !is.na(d2) & d2 <= 59                    ~ "VII_Ojo",
    p1 == "H" & !is.na(d2) & d2 >= 60 & d2 <= 95         ~ "VIII_Oido",
    p1 == "I"                                            ~ "IX_Circulatorio",
    p1 == "J"                                            ~ "X_Respiratorio",
    p1 == "K"                                            ~ "XI_Digestivo",
    p1 == "L"                                            ~ "XII_Piel",
    p1 == "M"                                            ~ "XIII_Musculoesqueletico",
    p1 == "N"                                            ~ "XIV_Genitourinario",
    p1 == "O"                                            ~ "XV_Embarazo",
    p1 == "P"                                            ~ "XVI_Perinatal",
    p1 == "Q"                                            ~ "XVII_Congenitas",
    p1 == "R"                                            ~ "XVIII_SintomasSignos",
    p1 %in% c("S","T")                                   ~ "XIX_Traumatismos",
    p1 %in% c("V","W","X","Y")                           ~ "XX_CausasExternas",
    p1 == "Z"                                            ~ "XXI_FactoresSalud",
    p1 == "U"                                            ~ "XXII_Especiales",
    TRUE                                                  ~ "OTRO"
  )
}

cat("=== Carga y preparación de datos ===\n")

df_raw <- read.csv(RUTA_CSV, stringsAsFactors = FALSE, na.strings = c("","NA")) |>
  mutate(EGRESO = factor(EGRESO, levels = c("Defuncion","Mejoria")),
         CIE10_CAP = mapear_cie10(CIE10))

df <- df_raw |>
  filter(GENERO != "N.E.") |>
  group_by(ESTADO) |> filter(n() >= 10) |> ungroup() |>
  mutate(PESO = if_else(ESTADO == "Sinaloa" & PESO == 99, NA_real_, PESO))

p1_p  <- quantile(df$PESO,   0.01, na.rm = TRUE)
p99_p <- quantile(df$PESO,   0.99, na.rm = TRUE)
p1_a  <- quantile(df$ALTURA, 0.01, na.rm = TRUE)
p99_a <- quantile(df$ALTURA, 0.99, na.rm = TRUE)

df <- df |>
  filter(is.na(PESO)   | between(PESO,   p1_p, p99_p)) |>
  filter(is.na(ALTURA) | between(ALTURA, p1_a, p99_a)) |>
  mutate(EDAD = if_else(EDAD > 120, NA_real_, EDAD))

p99_d <- quantile(df$DIAS_ESTANCIA, 0.99, na.rm = TRUE)
df <- df |> mutate(DIAS_ESTANCIA = if_else(DIAS_ESTANCIA > p99_d, NA_real_, DIAS_ESTANCIA))

df <- df |>
  group_by(CIE10_CAP, GENERO) |>
  mutate(PESO   = if_else(is.na(PESO),   median(PESO,   na.rm = TRUE), PESO),
         ALTURA = if_else(is.na(ALTURA), median(ALTURA, na.rm = TRUE), ALTURA)) |>
  ungroup()

med_peso   <- median(df$PESO,   na.rm = TRUE)
med_altura <- median(df$ALTURA, na.rm = TRUE)
df <- df |>
  mutate(PESO   = if_else(is.na(PESO),   med_peso,   PESO),
         ALTURA = if_else(is.na(ALTURA), med_altura, ALTURA))

df <- df |>
  mutate(REGION = case_when(
    ESTADO %in% c("Baja California","Baja California Sur","Chihuahua",
                  "Coahuila de Zaragoza","Durango","Nuevo Leon",
                  "Sinaloa","Sonora","Tamaulipas")                          ~ "REGION_I",
    ESTADO %in% c("Aguascalientes","Colima","Guanajuato","Jalisco",
                  "Michoacan de Ocampo","Nayarit","Queretaro de Arteaga",
                  "San Luis Potosi","Zacatecas")                            ~ "REGION_II",
    ESTADO %in% c("Distrito Federal","Mexico","Guerrero",
                  "Hidalgo","Morelos","Puebla","Tlaxcala")                  ~ "REGION_III",
    ESTADO %in% c("Campeche","Chiapas","Oaxaca","Quintana Roo",
                  "Tabasco","Veracruz de Ignacio de la Llave","Yucatan")    ~ "REGION_IV",
    ESTADO == "No Especificado"                                              ~ "NO_ESPECIFICADO",
    TRUE                                                                     ~ "OTRA"
  )) |>
  select(-ESTADO)

df <- df |>
  mutate(ASEGURADO_GRP = case_when(
    ASEGURADO == "IMSS"                                       ~ "IMSS",
    ASEGURADO == "ISSSTE"                                     ~ "ISSSTE",
    ASEGURADO %in% c("SEGURO POPULAR","SPSS")                ~ "SEGURO_POPULAR",
    ASEGURADO %in% c("SIN SEGURO","NINGUNA","NO ASEGURADO")  ~ "SIN_SEGURO",
    ASEGURADO == "SE IGNORA"                                  ~ "SE_IGNORA",
    TRUE                                                       ~ "OTRO"
  )) |>
  select(-ASEGURADO)

df <- df |>
  mutate(ACCIDENTE = if_else(ACCIDENTE == "NO APLICA", "NO", "SI"))

df <- df |>
  mutate(
    EGRESO        = factor(EGRESO, levels = c("Defuncion","Mejoria")),
    GENERO        = as.factor(GENERO),
    INFECCION     = as.factor(INFECCION),
    ACCIDENTE     = as.factor(ACCIDENTE),
    INDIGENA      = as.factor(INDIGENA),
    CIE10_CAP     = as.factor(CIE10_CAP),
    REGION        = as.factor(REGION),
    ASEGURADO_GRP = as.factor(ASEGURADO_GRP),
    MES_INGRESO   = factor(MES_INGRESO,
                           levels = c("Enero","Febrero","Marzo","Abril","Mayo",
                                      "Junio","Julio","Agosto","Septiembre",
                                      "Octubre","Noviembre","Diciembre"),
                           ordered = FALSE)
  ) |>
  select(-CIE10)

# ---- División CE/CV/CP, idéntica a la del qmd ----
set.seed(SEED)
idx_train <- createDataPartition(df$EGRESO, p = 0.60, list = FALSE)
df_train  <- df[ idx_train, ]
df_temp   <- df[-idx_train, ]
idx_val   <- createDataPartition(df_temp$EGRESO, p = 0.50, list = FALSE)
df_val    <- df_temp[ idx_val, ]
df_test   <- df_temp[-idx_val, ]

# ---- Imputación de NAs residuales en EDAD y DIAS_ESTANCIA ----
df_train$MES_INGRESO <- factor(df_train$MES_INGRESO, ordered = FALSE)
df_val$MES_INGRESO   <- factor(df_val$MES_INGRESO,   ordered = FALSE)
df_test$MES_INGRESO  <- factor(df_test$MES_INGRESO,  ordered = FALSE)

med_edad <- median(df_train$EDAD,          na.rm = TRUE)
med_dias <- median(df_train$DIAS_ESTANCIA, na.rm = TRUE)

for (nombre in c("df_train","df_val","df_test")) {
  tmp <- get(nombre)
  tmp$EDAD          <- ifelse(is.na(tmp$EDAD),          med_edad, tmp$EDAD)
  tmp$DIAS_ESTANCIA <- ifelse(is.na(tmp$DIAS_ESTANCIA), med_dias, tmp$DIAS_ESTANCIA)
  assign(nombre, tmp)
}

saveRDS(df_train, file.path(DIR_MODELOS, "df_train.rds"))
saveRDS(df_val,   file.path(DIR_MODELOS, "df_val.rds"))
saveRDS(df_test,  file.path(DIR_MODELOS, "df_test.rds"))

cat(sprintf("CE: %d | CV: %d | CP: %d\n", nrow(df_train), nrow(df_val), nrow(df_test)))

# ---- Cluster paralelo ----
cat(sprintf("\n=== Paralelismo: %d cores ===\n", N_CORES))
cl <- makePSOCKcluster(N_CORES)
registerDoParallel(cl)
on.exit(stopCluster(cl), add = TRUE)

ctrl_cv <- trainControl(
  method            = "cv",
  number            = 5,
  classProbs        = TRUE,
  summaryFunction   = twoClassSummary,
  sampling          = "up",
  savePredictions   = "final",
  allowParallel     = TRUE
)

tic <- function() Sys.time()
toc <- function(t0, etiqueta) {
  m <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  cat(sprintf("  %s -> %.1f min\n", etiqueta, m))
}

# ---- 1. kNN ----
cat("\n[1/8] kNN...\n"); t0 <- tic()
set.seed(SEED)
modelo_knn <- train(
  EGRESO ~ ., data = df_train,
  method = "knn",
  preProcess = c("center","scale"),
  tuneGrid = expand.grid(k = seq(3, 15, 2)),
  trControl = ctrl_cv,
  metric = "ROC"
)
saveRDS(modelo_knn, file.path(DIR_MODELOS, "modelo_knn.rds"))
toc(t0, "kNN")

# ---- 2. Naive Bayes (interfaz x/y para evitar mismatch de niveles en CV) ----
cat("\n[2/8] Naive Bayes...\n"); t0 <- tic()
predictores_train <- df_train[, setdiff(names(df_train), "EGRESO")]
set.seed(SEED)
modelo_nb <- train(
  x = predictores_train, y = df_train$EGRESO,
  method = "nb",
  tuneGrid = expand.grid(fL = c(0,1), usekernel = c(TRUE,FALSE), adjust = c(0.5,1.0)),
  trControl = ctrl_cv,
  metric = "ROC"
)
saveRDS(modelo_nb, file.path(DIR_MODELOS, "modelo_nb.rds"))
toc(t0, "NB")

# ---- 3. Árbol ----
cat("\n[3/8] Árbol...\n"); t0 <- tic()
set.seed(SEED)
modelo_arbol <- train(
  EGRESO ~ ., data = df_train,
  method = "rpart",
  tuneGrid = expand.grid(cp = c(0.0001, 0.0005, 0.001, 0.005, 0.01)),
  trControl = ctrl_cv,
  metric = "ROC"
)
saveRDS(modelo_arbol, file.path(DIR_MODELOS, "modelo_arbol.rds"))
toc(t0, "Árbol")

# ---- 4. Regresión Logística ----
cat("\n[4/8] Regresión Logística...\n"); t0 <- tic()
set.seed(SEED)
modelo_rl <- train(
  EGRESO ~ ., data = df_train,
  method = "glm", family = "binomial",
  preProcess = c("center","scale"),
  trControl = ctrl_cv,
  metric = "ROC"
)
saveRDS(modelo_rl, file.path(DIR_MODELOS, "modelo_rl.rds"))
toc(t0, "RL")

# ---- 5. SVM Radial (grid reducido por costo cuadrático) ----
cat("\n[5/8] SVM Radial...\n"); t0 <- tic()
set.seed(SEED)
modelo_svm <- train(
  EGRESO ~ ., data = df_train,
  method = "svmRadial",
  preProcess = c("center","scale"),
  tuneGrid = expand.grid(C = c(1, 10), sigma = 0.1),
  trControl = ctrl_cv,
  metric = "ROC"
)
saveRDS(modelo_svm, file.path(DIR_MODELOS, "modelo_svm.rds"))
toc(t0, "SVM")

# ---- 6. Red Neuronal ----
cat("\n[6/8] Red Neuronal...\n"); t0 <- tic()
set.seed(SEED)
modelo_nnet <- train(
  EGRESO ~ ., data = df_train,
  method = "nnet",
  preProcess = c("range"),
  tuneGrid = expand.grid(size = c(5, 10), decay = c(0.01, 0.1)),
  trControl = ctrl_cv,
  metric = "ROC",
  MaxNWts = 2000, maxit = 120, trace = FALSE
)
saveRDS(modelo_nnet, file.path(DIR_MODELOS, "modelo_nnet.rds"))
toc(t0, "NN")

# ---- 7. SVM refinado (Parte 3.7) ----
cat("\n[7/8] SVM refinado...\n"); t0 <- tic()
parametros_svm_fino <- expand.grid(
  C     = sort(unique(modelo_svm$bestTune$C * c(0.5, 2))),
  sigma = modelo_svm$bestTune$sigma
)
set.seed(SEED)
modelo_svm_fino <- train(
  EGRESO ~ ., data = df_train,
  method = "svmRadial",
  preProcess = c("center","scale"),
  tuneGrid = parametros_svm_fino,
  trControl = ctrl_cv,
  metric = "ROC"
)
saveRDS(modelo_svm_fino, file.path(DIR_MODELOS, "modelo_svm_fino.rds"))
toc(t0, "SVM-fino")

# ---- 8. Árbol refinado (Parte 3.7) ----
cat("\n[8/8] Árbol refinado...\n"); t0 <- tic()
parametros_arbol_fino <- expand.grid(
  cp = sort(unique(modelo_arbol$bestTune$cp * c(0.25, 0.5, 1, 2, 4)))
)
set.seed(SEED)
modelo_arbol_fino <- train(
  EGRESO ~ ., data = df_train,
  method = "rpart",
  tuneGrid = parametros_arbol_fino,
  trControl = ctrl_cv,
  metric = "ROC"
)
saveRDS(modelo_arbol_fino, file.path(DIR_MODELOS, "modelo_arbol_fino.rds"))
toc(t0, "Árbol-fino")

cat("\n=== Listo. Modelos en", DIR_MODELOS, "===\n")
cat("Hiperparámetros finales:\n")
cat(sprintf("  SVM:    C=%g, sigma=%g\n", modelo_svm$bestTune$C, modelo_svm$bestTune$sigma))
cat(sprintf("  SVM-fino: C=%g, sigma=%g\n",
            modelo_svm_fino$bestTune$C, modelo_svm_fino$bestTune$sigma))
cat(sprintf("  Árbol-fino: cp=%g\n", modelo_arbol_fino$bestTune$cp))
