

######################################


# 1. Extraemos datos
datos = read_excel("path/export_dataframe.xlsx", col_names = TRUE)
df = data.frame(datos)


# 2. Variables sobre asignaturas: Reemplazamos ',' por ' ' y creamos objetos document term matrix (vectorizamos)
df[,4] = gsub(",", " ", df[,4])
df[,5] = gsub(",", " ", df[,5])

it_train_4 = itoken(df[,4], 
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer, 
                  progressbar = TRUE)
vocab_4 = create_vocabulary(it_train_4)


vectorizer_4 = vocab_vectorizer(vocab_4)

dtm_train_4 = create_dtm(it_train_4, vectorizer_4)



it_train_5 = itoken(df[,5], 
                    preprocessor = tolower, 
                    tokenizer = word_tokenizer, 
                    progressbar = TRUE)
vocab_5 = create_vocabulary(it_train_5)


vectorizer_5 = vocab_vectorizer(vocab_5)

dtm_train_5 = create_dtm(it_train_5, vectorizer_5)

# 2. Quitamos variables originales y variables inútiles. Renombramos variables asignaturas
df_final = df[-c(3:5,56:57)]
dim(df_final)

cont = 53
for(i in colnames(dtm_train_4)){
  df_final = cbind(df_final,dtm_train_4[,i])
  colnames(df_final)[cont] = paste(i,'_fav', sep='')
  cont = cont + 1
}

cont = 66
for(i in colnames(dtm_train_5)){
  df_final = cbind(df_final,dtm_train_5[,i])
  colnames(df_final)[cont] = paste(i,'_hate', sep='')
  cont = cont + 1
}

# 3. Sustituimos valores NA por 3 (respuesta neutra)
df_final[is.na(df_final)] = 3



#-------------------- Árbol de decisión -------------------- 
library(readxl)
library(text2vec)

library(tree)
library(ggplot2) 
library(rpart) 
library(rpart.plot)
library(randomForest)

library(nnet)

attach(df_final)

# 4. Renombramos variables y creamos tabla de nombres originales y modificados
names(df_final)[1] = 'y'
nombres_var = matrix(NA, ncol=2, nrow = 78)
nombres_var[,1] = names(df_final)[3:80]

lista_renombres = numeric()
for(i in 1:78){
  lista_renombres[i] = paste("x",i, sep='')
}
nombres_var[,2] = lista_renombres


# 5. Quitamos variable grado de satisfacción
df_final = df_final[-c(2)]

# 6. Cambiamos todas las variables directamente
for(i in 2:79){
  names(df_final)[i] <- paste("x",i-1,sep='')
}

# 7. Convertimos variable a predecir en factor
df_final[,1] = as.factor(df_final[,1])

# 1. Modelo inicial
tree.inicial=tree(y~.,df_final) 
summary(tree.inicial)

# Vemos cuales son las primeras variables que diferencian

View(nombres_var)

"
62 -> matematicas fav
58 -> dibujo tecnico fav
61 -> biologia fav

"

# Predecimos en test
set.seed(1)
indices = sample(1:nrow(df_final), floor(0.25*nrow(df_final)))

tree.pred=predict(tree.inicial,df_final[indices,],type="class")

confu<-table(tree.pred,df_final[indices,1])
(confu[1,1]+confu[2,2]+confu[3,3]+confu[4,4])/34 # accuracy=0.882

# 2. Validación cruzada
cv.model=cv.tree(tree.inicial)

plot(cv.model$size,cv.model$dev,type='b') # el minimo está en 3

# Buscamos el mejor árbol con 3 nodos:
prune.boston=prune.tree(tree.inicial,best=3)




fit <- rpart(y ~ ., df_final, method = "class", cp=0)

# Vemos la exactitud en el conjunto test y dibujamos el árbol con rpart.plot
test = df_final[indices,]
preds = predict(fit, test, type = "class")
sum(preds == test$y)/nrow(test) # 0.705
rpart.plot(fit, type=1, extra = 102)

# Podamos el árbol usando el mejor valor de cp obtenido usando validación cruzada y pintamos el árbol.
pfit<- prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
rpart.plot(pfit, type=1, extra = 102)
preds = predict(pfit, test, type = "class")
sum(preds == test$y)/nrow(test) # acc = 0.794

# Creamos el árbol completo
fit <- rpart(y ~ ., df_final, method = "class", cp=-1)
rpart.plot(fit, type=1, extra = 102)



#-------------------- Random forest -------------------- 

# 0. Repetimos el preprocesamiento
################

names(df_final)[1] = 'y'
nombres_var = matrix(NA, ncol=2, nrow = 78)
nombres_var[,1] = names(df_final)[3:80]

lista_renombres = numeric()
for(i in 1:78){
  lista_renombres[i] = paste("x",i, sep='')
}
nombres_var[,2] = lista_renombres


# 2. Quitamos la variable grado de satisfacción
df_final = df_final[-c(2)]

# 3. Cambiamos todas las variables directamente
for(i in 2:79){
  names(df_final)[i] <- paste("x",i-1,sep='')
}


df_final[,1] = as.factor(df_final[,1])
##############

# 1. 


set.seed(1)
indices = sample(1:nrow(df_final), floor(0.75*nrow(df_final)))

train = df_final[indices,]
test = df_final[-indices,]
MSEi = numeric()

for(i in 1:dim(df_final)[2]){
  rf = randomForest(y~., data = train, ntree = 100, mtry=i)
  rf.pred = predict(rf, test, type="class")
  realidad = df_final[-indices,1]
  MSE = mean(rf.pred==realidad)
  MSEi[i]=MSE
}
which.min(MSEi) # el menor MSE de test se obtiene con 1 variable utilizada en cada corte

rf = randomForest(y ~ ., data = train, ntree = 100, mtry=1, importance=TRUE)
varImpPlot(rf, type=1)

# Las variables con más importancia son:

'
x61 -> biologia_fav
x30 -> No.tengo.mucha.imaginación
x41 -> No.me.importa.ser.el.centro.de.atención
x32 -> No.me.interesan.mucho.los.demás
x51 -> literatura_fav
'
rf_pred = predict(rf, test, type="class")
confu = table(rf_pred,df_final[-indices,1])

# El modelo predice siempre Matemáticas. Esto se debe a la falta de datos (desbalanceo en Matemáticas)
sum(diag(confu))/sum(confu) # acc=0.628


#-------------------- Red Neuronal -------------------- 

# 0. Repetimos el preprocesamiento
################

names(df_final)[1] = 'y'
nombres_var = matrix(NA, ncol=2, nrow = 78)
nombres_var[,1] = names(df_final)[3:80]

lista_renombres = numeric()
for(i in 1:78){
  lista_renombres[i] = paste("x",i, sep='')
}
nombres_var[,2] = lista_renombres


# 2. Quitamos la variable grado de satisfacción
df_final = df_final[-c(2)]

# 3. Cambiamos todas las variables directamente
for(i in 2:79){
  names(df_final)[i] <- paste("x",i-1,sep='')
}


df_final[,1] = as.factor(df_final[,1])
##############



set.seed(1)
indices = sample(1:nrow(df_final), floor(0.75*nrow(df_final)))
df_final
train = df_final[indices,]
test = df_final[-indices,]
length(test)
nn=neuralnet(y~.,data=train, hidden=3,act.fct = "logistic",
             linear.output = FALSE)
nn # te da los pesos,...
plot(nn) # los intercepts son los 1s

Predict=compute(nn,test)
Predict$net.result

prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred

pred <- predict(nn, test,type="class")

dim(pred)
prediction = ifelse(prob>0.5, 1, 0)
prediction[,1]  = ifelse(prediction[,1]==1, 'Matematicas')

length(df_final[-indices,1])

cm_nn <- table(pred=prediction, true=df_final[-indices,1])
cm_nn # matriz de confusion





accb = numeric()
for(b in 1:300){
  indices = sample(1:nrow(df_final), floor(0.75*nrow(df_final)))
  
  train = df_final[indices,]
  test = df_final[-indices,]
  
  nn <- nnet(y~ ., data=train, size=5, maxit=100, rang=0.1, decay=5e-4)
  pred <- predict(nn, test, type="class")
  cm_nn <- table(pred=pred, true=df_final[-indices,1])
  acc = sum(diag(cm_nn))/sum(cm_nn)
  accb[b]=acc
}
mean(accb) # 0.414

