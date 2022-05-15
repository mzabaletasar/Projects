rm(list = ls())

# Paquetes:
library(tseries)
library(forecast)
library(MASS)
library(dplyr)
library(tsoutliers)
library(data.table)

library(h2o)
library(bit64); h2o.init()

library(tsmp)

library(wmtsa)

library(plotly)
library(tidyverse)
library(timetk)
library(GeneCycle)
library(ggplot2)


# Función que te dice funciones y paquetes que usas:
# install.packages("NCmisc")
library(NCmisc)
list.functions.in.file(filename = 'code.R', alphabetic = TRUE)


setwd("path")
files <- list.files(pattern="*.txt")

pos_test <- numeric()
target_matrix <- matrix(NA, ncol = 2, nrow = 250)
for (i in 1:250){
  pos_test[i] <- as.integer(strsplit(files[i], "\\_")[[1]][5])
  
  begin <- as.integer(strsplit(files[i], "\\_")[[1]][6])
  b <- strsplit(files[i], "\\_")[[1]][7]
  end <- as.integer(substr(b,1,nchar(b)-4))
  L <- end - begin + 1
  target <- c(min(abs(begin - L), abs(begin - 100)), max(end + L, end + 100))
  target_matrix[i,] <- target
}
View(target_matrix)

len_series <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header=FALSE)
  len_series[i] <- dim(datos)[1]
}


# 1. ARIMA

result <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header=FALSE)
  datos$ID <- seq_along(datos[, 1])
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  datos$V1 <- datos$V1 + 1+abs(min(datos$V1))
  
  freq <- findfrequency(datos$V1)
  
  box_cox <- boxcox(V1 ~ ID,
                    data = datos,
                    lambda = c(0, 0.5, 1))
  lambda <- box_cox$x[which.max(box_cox$y)]

  if (1-lambda < 0.5-lambda) { 
    datos$target_transformado <- log(datos$V1)
  }
  else if (-lambda < 0.5-lambda) {
    datos$target_transformado <- datos$V1
  }
  else {
    datos$target_transformado <- (datos$V1)^lambda
  }
  
  datos.ts <- as.ts(datos$target_transformado, frequency = freq)
  
  ajuste_automatico <- auto.arima(datos.ts,
                                  max.d = 1, max.D = 1,
                                  max.p = 2, max.P = 2,
                                  max.q = 2, max.Q = 2, 
                                  seasonal = TRUE,
                                  ic = "aic",
                                  allowdrift = FALSE)
  lista_outliers <- locate.outliers(ajuste_automatico$residuals,
                                    pars = coefs2poly(ajuste_automatico),
                                    types = c("AO", "LS", "TC"), cval = 3)
  
  lista_outliers$abststat <- abs(lista_outliers$tstat)
  lista_outliers <- arrange(lista_outliers,desc(lista_outliers$abststat))
  
  outlier_pos <- lista_outliers[lista_outliers$ind >= pos_test[i],][1, "ind"]
  resultado <- (outlier_pos < target_matrix[i, 2]) & (outlier_pos > target_matrix[i, 1])
  total[i] <- resultado
}

acc <- mean(total)
acc # 0.405


# 2. Matrix Profile

# v1. Estándar
result <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header = FALSE)
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  model <- analyze(datos$V1, windows = 100)
  discords <- model$discord
  
  for (j in discords$discord_idx) {
    if (j >= pos_test[i]) {
      prediccion <- j
      break
    }
    else{
      prediccion <- 'no encuentra outlier en test'
    }
  }
  
  if ((prediccion <= target_matrix[i, 2]) & (prediccion >= target_matrix[i, 1])) {
    result[i] <- 1
  }
  else{
    result[i] <- 0
  }
}

acc <- mean(result)
acc # 0.505

# v2. Propio
result <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header = FALSE)
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  
  posiciones <- numeric()
  for (k in seq(100,200,5)){
    model <- analyze(datos$V1, windows = k)
    discords <- model$discord
    for (j in discords$discord_idx) {
      if (j >= pos_test[i]) {
        prediccion <- j
        break
      }
      else{
        prediccion <- 'no encuentra outlier en test'
      }
    }
    if (is.numeric(prediccion) == TRUE){
      posiciones <- append(posiciones, prediccion)
    }
  }
  if (length(posiciones) == 0) {
    result[i] <- 0
  }
  else {
    posiciones <- sort(posiciones)
    numero_outliers <- numeric()
    a = min(posiciones)
    b = max(posiciones)
    w = 0
    while (a+w*100 < b){
      numero_outliers = append(numero_outliers,
                               sum(posiciones >= a+w*100 & posiciones <= a + (w + 1)*100))
      w = w+1
      
    }
    
    indice <- which.max(numero_outliers)
    posiciones_validas <- posiciones[(posiciones >= a + (indice-1)*100) & (posiciones <= a + indice*100)]
    prediccion <- max(posiciones_validas)
    
    if ((prediccion <= target_matrix[i, 2]) & (prediccion >= target_matrix[i, 1])) {
      result[i] <- 1
    }
    else{
      result[i] <- 0
    }
  }
}

acc <- mean(result)
acc # 0.715


# 3. Wavelets
analisis_wavelet <- function(serie, numero_puntos, longitud_ventana){
  serie.dwt <- wavDWT(serie, n.levels = 6)
  MRD <- wavMRD(serie.dwt)
  D_1 <- MRD[["D1"]]
  
  Q_1 <- quantile(D_1, probs=0.25)
  Q_3 <- quantile(D_1, probs=0.75)
  sup_outliers <- Q_3 + 3*(Q_3 - Q_1)
  inf_outliers <- Q_1 - 3*(Q_3 - Q_1)
  
  D1_df <- as.data.frame(D_1) 
  D1_df <- cbind(D1_df, seq(1:length(serie)))
  resultado <- subset(D1_df, D_1 > sup_outliers | D_1 < inf_outliers)
  resultado <- arrange(resultado, desc(abs(D_1)))
  colnames(resultado) <- c("Valor","Observacion")
  
  contador_atipicos <- rep(-1, min(numero_puntos, length(resultado$Valor)))
  
  if (length(resultado$Valor) != 0) {
    for (i in 1:min(numero_puntos, length(resultado$Valor))) {
      for (j in 1:min(numero_puntos, length(resultado$Valor))) {
        if ((resultado$Observacion[i] <= resultado$Observacion[j] + longitud_ventana) 
            & (resultado$Observacion[i] >= resultado$Observacion[j] - longitud_ventana)) {
          contador_atipicos[i] <- contador_atipicos[i]+1
        }
      }
    }
  }

  if (length(resultado$Valor) != 0) {
    if (max(contador_atipicos) >= min(numero_puntos, length(resultado$Valor))/2) {
      return(resultado$Observacion[which.max(contador_atipicos)])
    }
    else{
      return(c("Dont know"))
    }
  }
  else{
    return(c("Dont know"))
  }
}

result <- numeric()
for (i in 1:200){
  x <- read.csv(files[i],header = FALSE)
  x <- x[,1]
  prediccion <- analisisWavelet(x,numeroPuntos=20,longitudVentana=200) 
  
  if (is.numeric(prediccion) == TRUE) {
    if ((prediccion <= target_matrix[i, 2]) & (prediccion >= target_matrix[i, 1])) {
      result <- append(result,1)
    }
    else{
      result <- append(result,0)
    }
  }
}
# Predigo en:
# 8 23 26 27 28 29 33 34 39 83 95 98 116 129 
# 130 131 133 134 135 136 137 141 142 147 190 191

result # [1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 1 1 1 0 0 1 0 1]
# Acierto: 8 23 27 28 29 39 83 95 98 116 129 130 131 133 135 136 137 147 191
# Fallo: 26 33 34 134 141 142 190

length(result) # 26
mean(result) # 0.73


# 4. iForest

# Accuracy v2: Entrenamos con todo, con contamination, predecimos test.
# Si la serie tiene más de 100.000 datos, fallamos

result <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header = FALSE)
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  
  if (dim(datos)[1] >= 100000){
    result[i] <- 0
  }
  
  else{
    train <- as.data.frame(datos[1 : (pos_test[i] - 1),]); colnames(train) <- c("V1")
    test <- as.data.frame(datos[pos_test[i] : dim(datos)[1],]); colnames(test) <- c("V1")
    datos.h2o <- as.h2o(datos)
    train.h2o <- as.h2o(train)
    test.h2o <- as.h2o(test)
    
    model <- h2o.isolationForest(x = "V1", training_frame = datos.h2o,
                                 seed = 1234, stopping_metric = "anomaly_score",
                                 score_each_iteration = TRUE,
                                 stopping_rounds = 3,
                                 sample_rate = 0.1,
                                 max_depth = 50, ntrees = 500, 
                                 contamination = 1/dim(datos)[1])
    
    score <- h2o.predict(model, test.h2o)
    result_pred <- as.vector(score$score)
    score_test <- cbind(RowScore = round(result_pred, 4), test)
    outlier <- score_test[which.max(score_test$RowScore),]
    posicion <- as.integer(row.names(outlier))
    posicion <- posicion + dim(train)[1]
    
    if ((posicion <= target_matrix[i, 2]) & (posicion >= target_matrix[i, 1])) {
      result[i] <- 1
    }
    else{
      result[i] <- 0
    }
  }
}

acc <- mean(result)
acc


# 5.1 Uniendo Matrix Profile propio con Wavelet

result_final <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header = FALSE)
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  
  x <- datos[,1]
  prediccion_wavelet <- analisis_wavelet(x, numeroPuntos=20, longitudVentana=200)
  
  posiciones <- numeric()
  for (k in seq(100,200,5)){
    model <- analyze(datos$V1, windows = k)
    discords <- model$discord
    for (j in discords$discord_idx) {
      if (j >= pos_test[i]) {
        prediccion <- j
        break
      }
      else{
        prediccion <- 'no encuentra outlier en test'
      }
    }
    if (is.numeric(prediccion) == TRUE){
      posiciones <- append(posiciones, prediccion)
    }
  }
  
  if (length(posiciones) == 0) {
    if (is.numeric(prediccion_wavelet) == TRUE){
      prediccion <- prediccion_wavelet
      if ((prediccion <= target_matrix[i, 2]) & (prediccion >= target_matrix[i, 1])) {
        result_final[i] <- 1
      }
      else{
        result_final[i] <- 0
      }
      
    }
    else {
      result_final[i] <- 0
    }
  }
    
  else {
    posiciones <- sort(posiciones)
    numero_outliers <- numeric()
    a = min(posiciones)
    b = max(posiciones)
    w = 0
    while (a+w*100 < b){
      numero_outliers = append(numero_outliers,
                               sum(posiciones >= a+w*100 & posiciones <= a + (w + 1)*100))
      w = w+1
    }
    
    indice <- which.max(numero_outliers)
    posiciones_validas <- posiciones[(posiciones >= a + (indice-1)*100) & (posiciones <= a + indice*100)]
    prediccion <- max(posiciones_validas)
    
    if (is.numeric(prediccion_wavelet) == TRUE) {
      prediccion <- prediccion_wavelet
    }
    
    if ((prediccion <= target_matrix[i, 2]) & (prediccion >= target_matrix[i, 1])) {
      result_final[i] <- 1
    }
    else{
      result_final[i] <- 0
    }
  }
  
}

acc_final <- mean(result_final)
acc_final # 0.72

# 5.2 Uniendo MP con Wavelet

result <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header = FALSE)
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  
  x <- datos[,1]
  prediccion_wavelet <- analisis_wavelet(x, numeroPuntos=20, longitudVentana=200)
  
  model <- analyze(datos$V1, windows = 100)
  discords <- model$discord
  
  for (j in discords$discord_idx) {
    if (j >= pos_test[i]) {
      prediccion <- j
      break
    }
    else{
      prediccion <- 'no encuentra outlier en test'
    }
  }
  if (is.numeric(prediccion_wavelet) == TRUE) {
    prediccion <- prediccion_wavelet
  }
  if ((prediccion <= target_matrix[i, 2]) & (prediccion >= target_matrix[i, 1])) {
    result[i] <- 1
  }
  else{
    result[i] <- 0
  }
}

acc <- mean(result)
acc # 0.54

# 5.3 Uniendo ARIMA con Wavelet

total <- numeric()
for (i in 1:200) {
  datos <- read.csv(files[i], header=FALSE)
  datos$ID <- seq_along(datos[, 1])
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  datos$V1 <- datos$V1 + 1+abs(min(datos$V1))
  
  x <- datos[,1]
  prediccion_wavelet <- analisis_wavelet(x, numeroPuntos=20, longitudVentana=200)
  
  freq <- findfrequency(datos$V1)
  
  box_cox <- boxcox(V1 ~ ID,
                    data = datos,
                    lambda = c(0, 0.5, 1))
  lambda <- box_cox$x[which.max(box_cox$y)]
  
  if (1-lambda < 0.5-lambda) { 
    datos$target_transformado <- log(datos$V1)
  }
  else if (-lambda < 0.5-lambda) {
    datos$target_transformado <- datos$V1
  }
  else {
    datos$target_transformado <- (datos$V1)^lambda
  }
  
  datos.ts <- as.ts(datos$target_transformado, frequency = freq)
  
  ajuste_Automatico <- auto.arima(datos.ts,
                                  max.d = 1, max.D = 1,
                                  max.p = 2, max.P = 2,
                                  max.q = 2, max.Q = 2, 
                                  seasonal = TRUE,
                                  ic = "aic",
                                  allowdrift = FALSE)
  lista_Outliers <- locate.outliers(ajuste_Automatico$residuals,
                                    pars = coefs2poly(ajuste_Automatico),
                                    types = c("AO","LS","TC"), cval = 3)
  
  lista_Outliers$abststat <- abs(lista_Outliers$tstat)
  lista_Outliers <- arrange(lista_Outliers,desc(lista_Outliers$abststat))
  
  outlier_pos <- lista_Outliers[lista_Outliers$ind >= pos_test[i],][1, "ind"]
  
  if (is.numeric(prediccion_wavelet) == TRUE) {
    outlier_pos <- prediccion_wavelet
  }
  
  resultado <- (outlier_pos < target_matrix[i, 2]) & (outlier_pos > target_matrix[i, 1])
  total[i] <- resultado
}

acc <- mean(total)
acc # 0.42





# data
setwd("path")
files <- list.files(path = "UCR_TimeSeriesAnomalyDatasets2021/UCR_Anomaly_FullData", pattern="*.txt")

# 1. plots periodogramas

for (j in c(1,15,5,6)) {
  datos <- read.csv(files[j], header=FALSE)
  for (i in 0:4) {
    mypath <- file.path("C:","Matemáticas","TFG","Time Series Anomaly Detection",
                        "images","plot periodos",
                        paste("plot", j,"_", i+1, ".jpg", sep = ""))
    periodo <- freqs[j]
    a <- periodo * i+1
    b <- periodo * (i+1)
    sub_plot <- plot_time_series(
      .data = data.frame(datos[a:b,]),
      .value = datos$V1[a:b],
      .date_var = 1:periodo,
      .interactive = FALSE)
    jpeg(filename = mypath)
    
    print(sub_plot + labs(title = as.character(j), subtitle = as.character(i+1)))
    dev.off()
  }
}

# 2. outliers

for (i in 1:25) {
  print(c("vuelta", i))
  datos <- read.csv(files[i], header=FALSE)
  datos$ID <- seq_along(datos[, 1])
  datos$V1 <- (datos$V1 - mean(datos$V1)) / sd(datos$V1)
  datos$V1 <- datos$V1 + 1+abs(min(datos$V1))
  
  box_cox <- boxcox(V1 ~ ID,
                    data = datos,
                    lambda = c(0, 0.5, 1))
  lambda <- box_cox$x[which.max(box_cox$y)]
  
  if (1-lambda < 0.5-lambda) {
    datos$target_transformado <- log(datos$V1)
  }
  else if (-lambda < 0.5-lambda) {
    datos$target_transformado <- datos$V1
  }
  else {
    datos$target_transformado <- (datos$V1)^lambda
  }
  
  datos.ts <- as.ts(datos$target_transformado, frequency = freq)
  
  ajuste_Automatico <- auto.arima(datos.ts,
                                  max.d = 1, max.D = 1,
                                  max.p = 2, max.P = 2,
                                  max.q = 2, max.Q = 2, 
                                  seasonal = TRUE,
                                  ic = "aic",
                                  allowdrift = FALSE)
  lista_Outliers <- locate.outliers(ajuste_Automatico$residuals,
                                    pars = coefs2poly(ajuste_Automatico),
                                    types = c("AO","LS","TC"), cval = 3)
  
  
  lista_Outliers$abststat <- abs(lista_Outliers$tstat)
  lista_Outliers <- arrange(lista_Outliers,desc(lista_Outliers$abststat))
  
  outlier_pos <- lista_Outliers[lista_Outliers$ind >= pos_test[i],][1,c(1:2)]
  
  if (outlier_pos$type == 'TC') {
    ts_plot <- plot_time_series(
      .data = datos,
      .value = datos$V1,
      .date_var = 1:dim(datos)[1],
      .interactive = FALSE)    
    print(ts_plot + 
            geom_vline(xintercept = pos_test[i], color = "red") + 
            geom_vline(xintercept = outlier_pos$ind, color = "yellow") +
            labs(title = as.character(i), subtitle = 'Temporary Changes'))
  }
  else if (outlier_pos$type == 'AO') {
    ts_plot <- plot_time_series(
      .data = datos,
      .value = datos$V1,
      .date_var = 1:dim(datos)[1],
      .interactive = FALSE)    
    print(ts_plot + 
            geom_vline(xintercept = pos_test[i], color = "red") + 
            geom_vline(xintercept = outlier_pos$ind, color = "orange") +
            labs(title = as.character(i), subtitle = 'Additive Outlier'))
  }
  else{
    ts_plot <- plot_time_series(
      .data = datos,
      .value = datos$V1,
      .date_var = 1:dim(datos)[1],
      .interactive = FALSE)    
    print(ts_plot + 
            geom_vline(xintercept = pos_test[i], color = "red") + 
            geom_vline(xintercept = outlier_pos$ind, color = "green") +
            labs(title = as.character(i), subtitle = 'Level Shift'))
  }
}

