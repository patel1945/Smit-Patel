setwd("F:\\CPTS_575\\Project")

library(data.table)
library(dplyr)
library(chron)
library(geosphere)
library(ROSE)
library(caret)
library(rpart)
library(caTools)
library(e1071)
library(randomForest)
library(DMwR)
library(cvms)
library(neuralnet)


df=fread("Final_Df_1.csv")
str(df)
dim(df)
summary(df)
df=df[df$Physical_Activity != "Unknown"]
df=df[df$Physical_Activity != "Lying"]
# Distribution of Physical Activity
prop.table(table(as.factor(df$`Physical_Activity`)))*100


#Re-Group
df1=df
df1$Physical_Activity=if_else(df1$Physical_Activity=="Biking"|df1$Physical_Activity=="Running","Running",df1$Physical_Activity)
df1$Physical_Activity=if_else(df1$Physical_Activity=="Standing"|df1$Physical_Activity=="Walking","Walking",df1$Physical_Activity)

# Distribution of Physical Activity after re-grouping 
prop.table(table(as.factor(df1$`Physical_Activity`)))



# With Only Accelerometer Data

#Response
y=as.factor(df1$Physical_Activity)
#Features
x=df[,5:14]

# Percentage of missing for each features 
tmp=apply(x,2,is.na)
(colSums(tmp)/nrow(x))*100

#Percentage of 0 for each features
tmp=apply(x,2,function(z) { sum(z==0)})
zero_pcnt=data.frame((tmp)/nrow(x))*100;zero_pcnt

#Remove variables with >90% zeroes
names=colnames(x)[zero_pcnt[,1]>90]
x1=select(x,-c(names))

data_all=data.frame("y"=y,x1)

set.seed(123)
split = sample.split(data_all$y, SplitRatio = 0.80)
training_set = subset(data_all, split == TRUE)
test_set = subset(data_all, split == FALSE)

#rpart
fit1 = rpart(y~. , training_set)
#Svm
fit2=svm(formula = y ~ .,data = training_set)
#Random Forest
fit3=randomForest(y~.,data = training_set)

#Train Accuracy

#Predict
pred_rpart = predict(fit1,training_set[,-1],type="class")
pred_svm=predict(fit2,training_set[,-1])
pred_rf=predict(fit3,training_set[,-1],type="class")


cm_rpart=table(training_set$y, pred_rpart)
Label_acc_rpart=(diag(cm_rpart)/rowSums(cm_rpart));Label_acc_rpart


cm_svm = table(training_set$y, pred_svm)
Label_acc_svm=(diag(cm_svm)/rowSums(cm_svm));Label_acc_svm

cm_rf = table(training_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf))*100;Label_acc_rf

accuracy_rpart = mean(training_set$y == pred_rpart);accuracy_rpart
accuracy_svm =mean(training_set$y == pred_svm);accuracy_svm
accuracy_rf=mean(training_set$y == pred_rf);accuracy_rf

#Test Accuracy

#Predict
pred_rpart = predict(fit1,test_set[,-1],type="class")
pred_svm=predict(fit2,test_set[,-1])
pred_rf=predict(fit3,test_set[,-1],type="class")


cm_rpart=data.frame(table(test_set$y, pred_rpart))
Label_acc_rpart=(diag(cm_rpart)/rowSums(cm_rpart));Label_acc_rpart


#Confusion Matrix Plot
cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_rpart,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Rpart")


cm_svm = table(test_set$y, pred_svm)
Label_acc_svm=(diag(cm_svm)/rowSums(cm_svm));Label_acc_svm
cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_svm,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Svm")


cm_rf = table(test_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf));Label_acc_rf

cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_rf,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Random Forest")




accuracy_rpart = mean(test_set$y == pred_rpart);accuracy_rpart
accuracy_svm =mean(test_set$y == pred_svm);accuracy_svm
accuracy_rf=mean(test_set$y == pred_rf);accuracy_rf


## With Both Accelerometer and GPS data

# With Only Accelerometer Data

#Response
y=as.factor(df1$Physical_Activity)
#Features
x=df[,5:20]

# Percentage of missing for each features 
tmp=apply(x,2,is.na)
(colSums(tmp)/nrow(x))*100

#Set missings as zero
x[is.na(x)]=0

#Percentage of 0 for each features
tmp=apply(x,2,function(z) { sum(z==0)})
zero_pcnt=data.frame((tmp)/nrow(x))*100;zero_pcnt

#Remove variables with >90% zeroes
names=colnames(x)[zero_pcnt[,1]>90]
x1=select(x,-c(names))

data_all=data.frame("y"=y,x1)

set.seed(123)
split = sample.split(data_all$y, SplitRatio = 0.80)
training_set = subset(data_all, split == TRUE)
test_set = subset(data_all, split == FALSE)

#rpart
fit1 = rpart(y~. , training_set)
#Svm
fit2=svm(formula = y ~ .,data = training_set)
#Random Forest
fit3=randomForest(y~.,data = training_set)

#Train Accuracy

#Predict
pred_rpart = predict(fit1,training_set[,-1],type="class")
pred_svm=predict(fit2,training_set[,-1])
pred_rf=predict(fit3,training_set[,-1],type="class")


cm_rpart=table(training_set$y, pred_rpart)
Label_acc_rpart=(diag(cm_rpart)/rowSums(cm_rpart));Label_acc_rpart


cm_svm = table(training_set$y, pred_svm)
Label_acc_svm=(diag(cm_svm)/rowSums(cm_svm));Label_acc_svm

cm_rf = table(training_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf))*100;Label_acc_rf

accuracy_rpart = mean(training_set$y == pred_rpart);accuracy_rpart
accuracy_svm =mean(training_set$y == pred_svm);accuracy_svm
accuracy_rf=mean(training_set$y == pred_rf);accuracy_rf

#Test Accuracy

#Predict
pred_rpart = predict(fit1,test_set[,-1],type="class")
pred_svm=predict(fit2,test_set[,-1])
pred_rf=predict(fit3,test_set[,-1],type="class")


cm_rpart=table(test_set$y, pred_rpart)
Label_acc_rpart=(diag(cm_rpart)/rowSums(cm_rpart));Label_acc_rpart


#Confusion Matrix Plot
cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_rpart,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Rpart")


cm_svm = table(test_set$y, pred_svm)
Label_acc_svm=(diag(cm_svm)/rowSums(cm_svm));Label_acc_svm
cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_svm,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Svm")


cm_rf = table(test_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf));Label_acc_rf

cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_rf,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Random Forest")


# Balanced Data 
prop.table(table(as.factor(data_all$y)))

obs_per_class=floor(nrow(data_all)/length(unique(data_all$y)))

#Over-sample Exercise
ex=data_all[data_all$y=="Exercise",]
idx=sample(nrow(ex),obs_per_class,replace = TRUE)
ex=ex[idx,]

#Over-sample Running
run=data_all[data_all$y=="Running",]
idx=sample(nrow(run),obs_per_class,replace = TRUE)
run=run[idx,]

#Under Sample Sitting
sit=data_all[data_all$y=="Sitting",]
idx=sample(nrow(run),obs_per_class,replace = TRUE)
sit=sit[idx,]

#Under Sampling Walking
walk=data_all[data_all$y=="Walking",]
idx=sample(nrow(walk),obs_per_class,replace = TRUE)
walk=walk[idx,]

data_all=rbind(ex,run,sit,walk)
prop.table(table(as.factor(data_all$y)))

set.seed(123)
split = sample.split(data_all$y, SplitRatio = 0.80)
training_set = subset(data_all, split == TRUE)
test_set = subset(data_all, split == FALSE)

#rpart
fit1 = rpart(y~. , training_set)
#Svm
fit2=svm(formula = y ~ .,data = training_set)
#Random Forest
fit3=randomForest(y~.,data = training_set)

#Train Accuracy

#Predict
pred_rpart = predict(fit1,training_set[,-1],type="class")
pred_svm=predict(fit2,training_set[,-1])
pred_rf=predict(fit3,training_set[,-1],type="class")

cm_rpart=table(training_set$y, pred_rpart)
Label_acc_rpart=(diag(cm_rpart)/rowSums(cm_rpart));Label_acc_rpart

cm_svm = table(training_set$y, pred_svm)
Label_acc_svm=(diag(cm_svm)/rowSums(cm_svm));Label_acc_svm

cm_rf = table(training_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf))*100;Label_acc_rf

accuracy_rpart = mean(training_set$y == pred_rpart);accuracy_rpart
accuracy_svm =mean(training_set$y == pred_svm);accuracy_svm
accuracy_rf=mean(training_set$y == pred_rf);accuracy_rf

#Test Accuracy

#Predict
pred_rpart = predict(fit1,test_set[,-1],type="class")
pred_svm=predict(fit2,test_set[,-1])
pred_rf=predict(fit3,test_set[,-1],type="class")


cm_rpart=table(test_set$y, pred_rpart)
Label_acc_rpart=(diag(cm_rpart)/rowSums(cm_rpart));Label_acc_rpart


#Confusion Matrix Plot
cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_rpart,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Rpart")


cm_svm = table(test_set$y, pred_svm)
Label_acc_svm=(diag(cm_svm)/rowSums(cm_svm));Label_acc_svm
cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_svm,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Svm")


cm_rf = table(test_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf));Label_acc_rf

cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_rf,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Random Forest")


accuracy_rpart = mean(test_set$y == pred_rpart);accuracy_rpart
accuracy_svm =mean(test_set$y == pred_svm);accuracy_svm
accuracy_rf=mean(test_set$y == pred_rf);accuracy_rf

# 5-Fold Cross Validation on Balanced Data and Using Random Forest Classifier
#Random Forest using Cross Validation
set.seed(123)
split = sample.split(data_all$y, SplitRatio = 0.80)
training_set = subset(data_all, split == TRUE)
test_set = subset(data_all, split == FALSE)

### Cross Validation :
control = trainControl(method="repeatedcv", number=5, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(training_set)-1)
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(y~., data=training_set, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)

#Train
pred_rf=predict(rf_default,training_set[,-1],type="raw")
cm_rf = table(training_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf));Label_acc_rf
accuracy_rf=mean(training_set$y == pred_rf);accuracy_rf

#Test
pred_rf=predict(rf_default,test_set[,-1],type="raw")
cm_rf = table(test_set$y, pred_rf)
Label_acc_rf=(diag(cm_rf)/rowSums(cm_rf));Label_acc_rf
accuracy_rf=mean(test_set$y == pred_rf);accuracy_rf


cm_dat=data.frame("target"=test_set$y,"Prediction"=pred_rf,
                  stringsAsFactors = FALSE)
cm_dat$target=as.character(cm_dat$target)
cm_dat$Prediction=as.character(cm_dat$Prediction)

eval <- evaluate(
  data = cm_dat,
  target_col = "target",
  prediction_cols = "Prediction",
  type = "multinomial"
)
plot_confusion_matrix(eval[["Confusion Matrix"]][[1]],add_row_percentages = FALSE,
                      add_col_percentages = FALSE)+ggtitle("Random Forest")




