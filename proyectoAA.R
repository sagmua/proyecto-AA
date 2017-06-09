## ----setup, include=FALSE------------------------------------------------
set.seed(5)
knitr::opts_chunk$set(echo = TRUE)
setwd("~/")

library("AppliedPredictiveModeling")

library("caret")
library("leaps")
library("glmnet")
library("readr")
library("ModelMetrics")
library("neuralnet")
library("e1071")
library("gbm")
library("plyr")
library("randomForest")


## ------------------------------------------------------------------------
datos_housing = as.data.frame( read_csv("datos/housing.data"))
head(datos_housing)

## ------------------------------------------------------------------------
train = sample (nrow(datos_housing), round(nrow(datos_housing)*0.8))
#definimos ambos conjuntos en dos data.frame diferentes:
housing_train = datos_housing[train,]
housing_test = datos_housing[-train,]

## ------------------------------------------------------------------------
pairs(datos_housing)

## ------------------------------------------------------------------------
housingTrans = preProcess(datos_housing[,-which(names(datos_housing) == "MEDV")], method = c("BoxCox", "center", "scale", "pca"), thresh = 0.9)
housingTrans$rotation

## ------------------------------------------------------------------------
nearZeroVar(housingTrans$rotation)

## ------------------------------------------------------------------------
housingTrans = preProcess(housing_train[,-which(names(housing_train) == "MEDV")],
                            method = c("BoxCox", "center", "scale"),thresh = 0.9)

housing_train[,-which(names(housing_train) == "MEDV")]=predict(housingTrans,
         housing_train[,-which(names(housing_train) == "MEDV")])

housing_test[,-which(names(housing_test) == "MEDV")] =predict(housingTrans,housing_test[,-which(names(housing_test) == "MEDV")])

## ------------------------------------------------------------------------
regsub_housing =regsubsets(datos_housing[,-which(names(datos_housing) == "MEDV")],
                             datos_housing[,which(names(datos_housing) == "MEDV")])

summary(regsub_housing)

## ------------------------------------------------------------------------
etiquetas = housing_train[,which(names(housing_train) == "MEDV")]
tr = housing_train[,-which(names(housing_train) == "MEDV")]
tr = as.matrix(tr)
crossvalidation =cv.glmnet(tr,etiquetas,alpha=0)
print(crossvalidation$lambda.min)

## ------------------------------------------------------------------------
modelo_reg = glmnet(tr,etiquetas,alpha=0,lambda=crossvalidation$lambda.min)
print(modelo_reg)

## ------------------------------------------------------------------------
modelo_reg = glmnet(tr,etiquetas,alpha=0,lambda=0)
print(modelo_reg)

## ------------------------------------------------------------------------
m_muestra_housing = lm(MEDV ~ LSTAT, data=housing_train)

## ----calculoMSE----------------------------------------------------------
calculoMSE  = function (modelo, test, variable_respuesta){
  prob_test = predict(modelo, test[,-which(names(test) == variable_respuesta)])

  mse(test[,which(names(test) == variable_respuesta)], prob_test)
}

## ------------------------------------------------------------------------
etest_mmuestra = calculoMSE(m_muestra_housing, housing_test, "MEDV")
etest_mmuestra

## ------------------------------------------------------------------------
m1_housing = lm(MEDV ~ LSTAT + RM, data=housing_train)

etest_m1 = calculoMSE(m1_housing, housing_test, "MEDV")
etest_m1

## ------------------------------------------------------------------------
m2_housing = lm(MEDV ~ LSTAT + RM + PTRATIO, data=housing_train)

etest_m2 = calculoMSE(m2_housing, housing_test, "MEDV")
etest_m2

## ------------------------------------------------------------------------
m3_housing = lm(MEDV ~ LSTAT + RM + PTRATIO + DIS, data=housing_train)

etest_m3 = calculoMSE(m3_housing, housing_test, "MEDV")
etest_m3

## ------------------------------------------------------------------------
m4_housing = lm(MEDV ~ LSTAT + RM + PTRATIO + DIS + NOX, data=housing_train)

etest_m4 = calculoMSE(m4_housing, housing_test, "MEDV")
etest_m4

## ------------------------------------------------------------------------
m5_housing = lm(MEDV ~ LSTAT + RM + I(PTRATIO^2) + DIS, data=housing_train)

etest_m5 = calculoMSE(m5_housing, housing_test, "MEDV")
etest_m5

## ------------------------------------------------------------------------
m6_housing = lm( MEDV ~ LSTAT * RM * PTRATIO * DIS, data=housing_train)

etest_m6 = calculoMSE(m6_housing, housing_test, "MEDV")
etest_m6

## ------------------------------------------------------------------------
crossValidationNN = function(datos, capas){
  maxs = apply(datos, 2, max)
  mins = apply(datos, 2, min)
  datos = as.data.frame(scale(datos, center = mins, scale = maxs - mins))
  set.seed(6)
  #realizamos 5 particiones:
  folds = suppressWarnings(split(datos, sample(rep(1:5, nrow(datos)/5))))
  errores = as.numeric()
  for (i in 1:5){
    test_ = folds[[i]]
    train_ = data.frame()
    for (j in 1:5){
      if(j!=i)
        train_ = rbind(train_, folds[[j]])
    }

    n = names(train_)
    f = as.formula(paste("MEDV ~", paste(n[!n %in% "MEDV"], collapse = " + ")))
    nn = neuralnet(f,data=train_,hidden=capas,linear.output=T)
    pr.nn = compute(nn,test_[,1:13])
    pr.nn_ = pr.nn$net.result*(max(datos_housing$MEDV)-min(datos_housing$MEDV))+min(datos_housing$MEDV)
    test.r = (test_$MEDV)*(max(datos_housing$MEDV)-min(datos_housing$MEDV))+min(datos_housing$MEDV)
    MSE.nn = sum((test.r - pr.nn_)^2)/nrow(test_)

    errores = c(errores, MSE.nn)
  }

  mean(errores)
}

eout_nn_1_5=crossValidationNN(datos_housing,5)
eout_nn_1_5
eout_nn_1_3=crossValidationNN(datos_housing,3)
eout_nn_1_3
eout_nn_2_53=crossValidationNN(datos_housing,c(5,3))
eout_nn_2_53
eout_nn_2_74=crossValidationNN(datos_housing,c(7,4))
eout_nn_2_74
eout_nn_3_853=crossValidationNN(datos_housing,c(8,5,3))
eout_nn_3_853
eout_nn_3_742=crossValidationNN(datos_housing,c(7,4,2))
eout_nn_3_742

## ------------------------------------------------------------------------
maxs = apply(datos_housing, 2, max)
mins = apply(datos_housing, 2, min)
datos_escalados = as.data.frame(scale(datos_housing, center = mins, scale = maxs - mins))
housing_train_esc = datos_escalados[train,]
n = names(housing_train_esc)
f = as.formula(paste("MEDV ~", paste(n[!n %in% "MEDV"], collapse = " + ")))
nn_winner = neuralnet(f,data=housing_train_esc,hidden=c(7,4,2),linear.output=T)
plot(nn_winner)

## ------------------------------------------------------------------------
tune_gamma_svm = tune(svm, MEDV ~ ., data=housing_train,
                      ranges = list(gamma=seq(0,1,0.01)))

tune_gamma_svm

## ------------------------------------------------------------------------
crossValidationSVM = function(datos){
  set.seed(6)
  #realizamos 5 particiones:
  folds = suppressWarnings(split(datos, sample(rep(1:5, nrow(datos)/5))))
  errores = as.numeric()
  for (i in 1:5){
    test_ = folds[[i]]
    train_ = data.frame()
    for (j in 1:5){
      if(j!=i)
        train_ = rbind(train_, folds[[j]])
    }

    svm_housing = svm(MEDV ~ ., data=train_, kernel="radial", gamma=0.09)
    pred_svm = predict(svm_housing, test_)
    error = mse(test_$MEDV, pred_svm)

    errores = c(errores, error)
  }

  mean(errores)
}

crossValidationSVM(datos_housing)

## ------------------------------------------------------------------------
grid_gbm = expand.grid(.interaction.depth = (1:5)*2, .n.trees = (1:10)*50, .shrinkage = 0.1, .n.minobsinnode=10)

bootstrap_control = trainControl(number = 200)
gbm_housing = train(housing_train[,1:13], housing_train[,14],
                    method="gbm", trControl = bootstrap_control,
                    verbose=F, bag.fraction = 1,
                    tuneGrid = grid_gbm)

gbm_housing$bestTune

## ------------------------------------------------------------------------
set.seed(7)
boosting_housing = gbm(MEDV ~ ., data = housing_train,
                       n.trees = 350, interaction.depth = 4,
                       shrinkage = 0.1, n.minobsinnode = 10,
                       cv.folds = 5)

pred_boosting = predict(boosting_housing, housing_test,
                        n.trees = 350, interaction.depth = 4,
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

mse(housing_test$MEDV, pred_boosting)

## ------------------------------------------------------------------------
set.seed(6)
tune_rf = tune(randomForest, MEDV ~ ., data=housing_train,
               ranges = list(ntree=seq(25,500,25)))

tune_rf

## ------------------------------------------------------------------------
crossValidationRF = function(datos){
  set.seed(6)
  #realizamos 5 particiones:
  folds = suppressWarnings(split(datos, sample(rep(1:5, nrow(datos)/5))))
  errores = as.numeric()
  m = ncol(housing_train)/3
  for (i in 1:5){
    test_ = folds[[i]]
    train_ = data.frame()
    for (j in 1:5){
      if(j!=i)
        train_ = rbind(train_, folds[[j]])
    }

    rf_housing = randomForest(MEDV ~ ., train_, ntree=50, mtry=m)
    error = calculoMSE(rf_housing, test_, "MEDV")

    errores = c(errores, error)
  }

  mean(errores)
}

eout_rf = crossValidationRF(datos_housing)
eout_rf

