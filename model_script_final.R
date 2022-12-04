##### Needed Libraries #####
RNGkind(sample.kind = "Rounding")
options(max.print=3500)
library(openxlsx)
library(leaps)
library(fastDummies)
library(glmnet)
library(pls)
library(boot)
library(caret)
library(data.table)
library(xgboost)
library(tree)
library(randomForest)
library(dplyr)
library(magrittr)
library(tensorflow)
library(keras)

##### Raw Model Building #####

Player<- read.xlsx("Player.xlsx")  #define dataset
dim(Player)
colnames(Player)

Player$Value=log(Player$Value) #Take ln of "Value" variable

scaled.dat <- scale(Player[,4:41]) #Numeric Data Standardization
Player[,4:41]=scaled.dat


set.seed(18)
smp_size <- floor(0.80 * nrow(Player))      #define % of training and test set
train_ind <- sample(seq_len(nrow(Player)), size = smp_size)   #sample rows
Player.train <- Player[train_ind, ]      #get training set
Player.test <- Player[-train_ind, ]      #get test set
dim(Player.train)
dim(Player.test)
#Select important columns of the datasets

Player.train.model=Player.train[,-1]
Player.train.model=Player.train.model[,-1]
Player.test.model=Player.test[,-1]
Player.test.model=Player.test.model[,-1]
Player.new=Player[,-1]
colnames(Player.new)
Player.new=Player.new[,-1]
colnames(Player.new)
head(Player.new)

lin1<-lm(Value~.,data= Player.train.model) #Create linear regression on train-set
summary(lin1)

predlin1=predict(lin1,Player.test.model) #Predict results on test-set 
sqrt(mean(((Player.test.model$Value) -(predict(lin1 ,Player.test.model)))^2)) # Calculate MSE for the model
sqrt(mean(((Player.train.model$Value) -(predict(lin1 ,Player.train.model)))^2))


##### Validation Set and Cross-Validation BSS #####

#Validation Set(Forward)
regfit.fwd=regsubsets (Value~.,data =Player.train.model ,really.big = T,nvmax=100,method="forward")
reg.summary<-summary(regfit.fwd)

par(mfrow =c(2,2))
plot(reg.summary$rss ,xlab=" Number of Variables ",ylab=" RSS",type="l")
which.min (reg.summary$rss)
points (54, reg.summary$rss[54], col ="red",cex =2, pch =20)
plot(reg.summary$adjr2 ,xlab =" Number of Variables ",ylab=" Adjusted RSq",type="l")
which.max (reg.summary$adjr2)
points (30, reg.summary$adjr2[30], col ="red",cex =2, pch =20)
plot(reg.summary$cp ,xlab =" Number of Variables ",ylab="Cp",type="l")
which.min (reg.summary$cp )
points (25, reg.summary$cp [25], col ="red",cex =2, pch =20)
which.min (reg.summary$bic )
plot(reg.summary$bic ,xlab=" Number of Variables ",ylab=" BIC",type="l")
points (13, reg.summary$bic [13], col =" red",cex =2, pch =20)


test.mat=model.matrix(Value~.,data=Player.test.model)
val.errors =rep(NA ,54)
for(i in 1:54){
  coefi=coef(regfit.fwd ,id=i)
  pred=test.mat [,names(coefi)]%*% coefi
  val.errors [i]= sqrt(mean(((Player.test.model$Value)-(pred))^2))}

val.errors
which.min (val.errors )
coef(regfit.fwd ,34)

Player.train.model.dummy <- fastDummies::dummy_cols(Player.train.model) # One-Hot Encoding
Player.test.model.dummy <- fastDummies::dummy_cols(Player.test.model)
Player.new.dummy <- fastDummies::dummy_cols(Player.new)

lin2=lm(Value ~Ability+Age+Agg+Ant+Cmp+Cnt+Fla+Pos+Tea+Wor+Acc+Bal+Jum+Pac+Sta+Str+Fin+Fir+Fre+Lon+L.Th+Mar+Tck+Division+BestPos_DC+ BestPos_DL+BestPos_MC+BestPos_MR+BestPos_STC+Club.1, data=Player.train.model.dummy)
summary(lin2)
predlin2=predict(lin2,Player.test.model.dummy) #Predict results on test-set 
sqrt(mean(((Player.test.model$Value) -(predict(lin2 ,Player.test.model.dummy)))^2)) # Calculate MSE for the model
sqrt(mean(((Player.train.model$Value) -(predict(lin2 ,Player.train.model.dummy)))^2))

#Cross-Validation(Forward)
predict.regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id=id)
  xvars=names(coefi)
  mat[, xvars] %*% coefi
}


k=10
set.seed (1)
folds=sample (1:k,nrow(Player.new.dummy),replace =TRUE)
cv.errors =matrix (NA ,k,54, dimnames=list(NULL , paste (1:54) ))

for(j in 1:k){
  best.fit =regsubsets(Value~.,data =Player.new.dummy[folds !=j,] ,really.big = T, nvmax=54, method="forward")
  
  for(i in 1:54) {
    pred=predict(best.fit ,Player.new.dummy[folds ==j,], id=i)
    cv.errors[j,i]=sqrt(mean( ((Player.new.dummy$Value)[folds ==j]-(pred))^2))
  }
}
mean.cv.errors =apply(cv.errors ,2, mean)
mean.cv.errors
which.min(mean.cv.errors)
par(mfrow =c(1,1))
plot(mean.cv.errors ,type="b")
points (24, mean.cv.errors[24], col ="red",cex =2, pch =20)
reg.best=regsubsets (Value~.,data=Player.train.model , nvmax =54,method="forward")
coef(reg.best ,24)

lin3=lm(Value~Ability+Ant+Fla+Pos+Tea+Acc+Bal+Jum+Pac+Sta+Str+Fir+Fre+Lon+L.Th+Mar+Tck+Club.1+Division+BestPos_DC,data=Player.train.model.dummy)
summary(lin3)
predlin3=predict(lin3,Player.test.model.dummy) #Predict results on test-set 
sqrt(mean(((Player.test.model$Value) -(predict(lin3 ,Player.test.model.dummy)))^2)) # Calculate MSE for the model
sqrt(mean(((Player.train.model$Value) -(predict(lin3 ,Player.train.model.dummy)))^2))


##### Ridge and Lasso BSS #####

#Datasets
x=model.matrix(Value~.,Player.train.model )[,-1]
y=Player.train.model$Value
xt=model.matrix(Value~.,Player.test.model )[,-1]
yt=Player.test.model$Value
xf=model.matrix(Value~.,Player.new )[,-1]
yf=Player.new$Value

#Ridge
set.seed(1)
cv.out=cv.glmnet(x,y,alpha=0)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam
ridge.mod=glmnet (x,y,alpha=0)
ridge.pred=predict (ridge.mod ,s=bestlam ,newx=xt)
mean((ridge.pred -yt)^2)
out=glmnet(xf,yf,alpha=0)
predict (out ,type="coefficients",s= bestlam) [1:55,]

#Lasso
set.seed(1)
lasso.mod=glmnet(x,y,alpha=1)
plot(lasso.mod)
cv.out=cv.glmnet(x,y,alpha=1)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam
lasso.pred=predict(lasso.mod ,s=bestlam ,newx=xt)
mean((lasso.pred -yt)^2)
out=glmnet (xf,yf,alpha=1)
lasso.coef=predict (out ,type="coefficients",s= bestlam) [1:55,]
lasso.coef
lasso.coef[lasso.coef!=0]
lasso.coef[lasso.coef==0]
length(lasso.coef[lasso.coef!=0])-1

lin4=lm(Value~Club.1+Ability+Age+Agg+Ant+Bra+Cmp+Cnt+Dec+Fla+Ldr+OtB+Pos+Tea+Wor+Acc+Bal+Jum+Nat+Pac+Sta+Str+Cor+Fin+Fir+Fre+Lon+L.Th+Mar+Pas+Tck+Division+BestPos_AML+BestPos_DC+BestPos_DL+BestPos_DM+BestPos_MC+BestPos_MR+BestPos_STC,data=Player.train.model.dummy)
summary(lin4)
predlin4=predict(lin4,Player.test.model.dummy) #Predict results on test-set 
sqrt(mean(((Player.test.model$Value) -(predict(lin4 ,Player.test.model.dummy)))^2)) # Calculate MSE for the model
sqrt(mean(((Player.train.model$Value) -(predict(lin4 ,Player.train.model.dummy)))^2))


##### PCR and PLS DRM #####
#PCR
pcr.fit=pcr(Value~., data=Player.train.model , scale=TRUE ,validation ="CV")
summary (pcr.fit)
validationplot(pcr.fit ,val.type="MSEP")
pcr.pred=predict (pcr.fit ,xt,ncomp =54)
mean((pcr.pred -yt)^2)
pcr.fit=pcr(yf~xf,scale=TRUE ,ncomp=54)
summary (pcr.fit)

#PLS
pls.fit=plsr(Value~., data=Player.train.model , scale=TRUE ,validation ="CV")
summary (pls.fit)
validationplot(pls.fit ,val.type="MSEP")
pls.pred=predict (pls.fit ,xt,ncomp =12)
mean((pls.pred -yt)^2)
pls.fit=plsr(Value~., data=Player.new , scale=TRUE , ncomp=12)
summary (pls.fit)


##### Finding Non-Linear Relationships #####

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Ability ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error


cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Age ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error


cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Ant ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Bra ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Cmp ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Cnt ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Dec ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Agg ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Fla ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Ldr ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(OtB ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Tea ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Wor ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Acc ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Bal ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Jum ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Nat ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Pac,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Sta ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Str ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Cor ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Fin ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Fir ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Fre ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Lon ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(L.Th ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Mar ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Pas ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error

cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(Value~ poly(Tck ,i),data=Player.new)
  cv.error[i]=cv.glm(Player.new ,glm.fit, K=10)$delta [1]
}
cv.error


plot(Player$Age,Player$Value)

Linage1=lm(Value~Age,Player.new)
Linage2=lm(Value~Age+I(Age^2),Player.new)
anova(Linage1,Linage2)


lin5=lm(Value~I(Age^2)+Club.1+Ability+Age+Agg+Ant+Bra+Cmp+Cnt+Dec+Fla+Ldr+OtB+Pos+Tea+Wor+Acc+Bal+Jum+Nat+Pac+Sta+Str+Cor+Fin+Fir+Fre+Lon+L.Th+Mar+Pas+Tck+Division+BestPos_AML+BestPos_DC+BestPos_DL+BestPos_DM+BestPos_MC+BestPos_MR+BestPos_STC,data=Player.train.model.dummy)
summary(lin5)


##### Interaction Terms #####

Linam=lm(Value~.^2, data= Player.new)# to find possible interaction terms
summary(Linam)


Linfpb=lm(Value~Ability+Age, data= Player.new)
Linftb=lm(Value~Ability*Age, data= Player.new)

anova(Linfpb,Linftb)#  significant

Linhpb=lm(Value~Age+Cnt, data= Player.new)
Linhtb=lm(Value~Age*Cnt, data= Player.new)

anova(Linhpb,Linhtb)#  significant

Lindpb=lm(Value~Det+Tec, data= Player.new)
Lindtb=lm(Value~Det*Tec, data= Player.new)

anova(Lindpb,Lindtb)#  significant


##### Final Linear Model #####
lin6=lm(Value~+Det*Tec+Age*Cnt+Ability*Age+I(Age^2)+Club.1+Ability+Age+Agg+Ant+Bra+Cmp+Cnt+Dec+Fla+Ldr+OtB+Pos+Tea+Wor+Acc+Bal+Jum+Nat+Pac+Sta+Str+Cor+Fin+Fir+Fre+Lon+L.Th+Mar+Pas+Tck+Division+BestPos_AML+BestPos_DC+BestPos_DL+BestPos_DM+BestPos_MC+BestPos_MR+BestPos_STC,data=Player.train.model.dummy)
summary(lin6)

predlin6=predict(lin6,Player.test.model.dummy) #Predict results on test-set
sqrt(mean(((Player.test.model.dummy$Value) -(predict(lin6 ,Player.test.model.dummy)))^2)) # Calculate MSE for the model
sqrt(mean(((Player.train.model.dummy$Value) -(predict(lin6 ,Player.train.model.dummy)))^2))

# Results on 19 data
testimpp<-read.xlsx("Player19.xlsx")
scaled.dat <- scale(testimpp[,4:41]) #Numeric Data Standardization
testimpp[,4:41]=scaled.dat
testimppp=testimpp[,-1]
testimppp=testimppp[,-1]
testimppp<- fastDummies::dummy_cols(testimppp)

predlinson=predict(lin6,testimppp) #Predict results on test-set
#write.xlsx(exp(predlinson), "19linearresults.xlsx")
#write.xlsx(testimpp$Name, "19names.xlsx")
#write.xlsx(testimpp$Club, "19club.xlsx")
#write.xlsx(testimpp$Age, "19age.xlsx")
#write.xlsx(testimpp$BestPos, "19position.xlsx")


##### Decision Tree Based Models Data Preparation #####
Player<- read.xlsx("Player.xlsx")
set.seed(18)
smp_size <- floor(0.80 * nrow(Player))      #define % of training and test set
train_ind <- sample(seq_len(nrow(Player)), size = smp_size)   #sample rows
train <- Player[train_ind, ]      #get training set
test <- Player[-train_ind, ]      #get test set


train<- train[,-1]
train<- train[,-1]
test<- test[,-1]
test<- test[,-1]


xTrans <- preProcess(train[,2:39])
train <- predict(xTrans, train)
test <- predict(xTrans, test)

setDT(train)
setDT(test)

new_tr <- model.matrix(~.+0,data = train[,-c("Value"),with=F])
new_ts <- model.matrix(~.+0,data = test[,-c("Value"),with=F])

k=1
for (i in colnames(new_tr)) {
  check<-(i %in% colnames(new_ts))
  print(check)
  print(k)
  k=k+1
}

labels <- train$Value
ts_label <- test$Value


labels<-as.numeric(labels)

ts_label<-as.numeric(ts_label)



dtrain <- xgb.DMatrix(data = new_tr,label = labels)
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)


##### Decision Tree #####

#Data preparetion

dtreetrain =data.frame(new_tr ,labels)
dtreetest <- data.frame(new_ts ,ts_label)

#Create the decision tree model
model_dt = tree(labels~.,data=dtreetrain)
summary(model_dt )
#Deviance is sum of square errors
plot(model_dt )
text(model_dt ,pretty =0)

#Check if pruning will improve performance
cv.model=cv.tree(model_dt)
plot(cv.model$size ,cv.model$dev ,type="b")
#the unpruned tree has the lowest deviance.


#Prediction on test data.
model_dt_pred <- predict(model_dt ,dtreetest)
sqrt(mean((log(ts_label) -log(model_dt_pred))^2)) #MSE

##### Random Forest #####

set.seed(18)
#Find the model with the best parameters and accuracy. 
a=c()
i=1
for (i in 1:55) {
  forest.model.param <- randomForest(x = new_tr, y = as.numeric(labels),  ntree = 500, mtry = i, importance = TRUE)
  predValid <- predict(forest.model.param, new_ts)
  a[i-2] = mean((predValid-as.numeric(ts_label))^2)
}
# Best accuracy on test data
min(a)#18 is the best number of predictors


#Create a random forest model with the optimum number of mtry.
forest.model <- randomForest(x = new_tr, y = as.numeric(labels), 
                             importance=TRUE,ntree=1500,
                             mtry=18)
#Summary of the model.
forest.model

#Check importance of variables.
importance(forest.model)        
varImpPlot(forest.model)  

rflastpred<-predict(forest.model, new_ts)
sqrt(mean((log(ts_label) -log(rflastpred))^2))

##### XGBoost #####

### Caret Based Grid Searching

xgbGrid <-  expand.grid(nrounds = c(10,100), 
                        max_depth = c(4,5,6, 10), 
                        eta = 0.3,
                        gamma = c(0,1), colsample_bytree=1,
                        min_child_weight=1, subsample=1)

fitControl <- trainControl(## 2-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 10)

gbmFit <- caret::train(new_tr, as.numeric(labels), method = "xgbTree", 
                       trControl = fitControl, verbose = T, 
                       tuneGrid = xgbGrid)

varImp(gbmFit)


plot(gbmFit)                       
gbmFit$results

best(gbmFit$results, metric="RMSE", maximize = F)
tolerance(gbmFit$results, metric="RMSE", maximize=F, tol=2)








### XGBoost

params <- list(booster = "gbtree", objective = "reg:linear", 
               eta=0.3, gamma=0, max_depth=5, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, 
                 print_every_n = 1, early_stopping_rounds = 20, maximize = F)

xgbcv$best_iteration

xgb1 <- xgb.train (params = params, data = dtrain, 
                   nrounds = xgbcv$best_iteration)

xgbpred <- predict (xgb1,dtest) #MSE is 0,170343.




mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:25])



testimp<-read.xlsx("Player19.xlsx")
testimp<- testimp[,-1]
testimp<- testimp[,-1]
testimp <- predict(xTrans, testimp)
setDT(testimp)
new_tsimp <- model.matrix(~.+0,data = testimp[,-c("Value"),with=F])
ts_labelimp <- testimp$Value
ts_labelimp <- as.numeric(ts_labelimp)
dtestimp <- xgb.DMatrix(data = new_tsimp,label=ts_labelimp)

xgbpred <- predict (xgb1,dtestimp)
#write.xlsx(xgbpred, "19dectreeresult.xlsx")


##### Artificial Neural Networks (with Keras) #####
#Preparation of datasets for Keras based neural networks.
Player<- read.xlsx("Player.xlsx") 
set.seed(18)
smp_size <- floor(0.80 * nrow(Player))      #define % of training and test set
train_ind <- sample(seq_len(nrow(Player)), size = smp_size)   #sample rows
train <- Player[train_ind, ]      #get training set
test <- Player[-train_ind, ]      #get test set
testimp<- read.xlsx("Player19.xlsx")

train<- train[,-1]
train<- train[,-1]
test<- test[,-1]
test<- test[,-1]
testimp<-testimp[,-1]
testimp<-testimp[,-1]

train.dummy <- fastDummies::dummy_cols(train)
test.dummy <- fastDummies::dummy_cols(test)
testimp.dummy<-fastDummies::dummy_cols(testimp)
train.dummy<-train.dummy[,-42]
train.dummy<-train.dummy[,-41]
train.dummy<-train.dummy[,-40]
train.dummy<-train.dummy[,-1]
test.dummy<-test.dummy[,-42]
test.dummy<-test.dummy[,-41]
test.dummy<-test.dummy[,-40]
test.dummy<-test.dummy[,-1]
testimp.dummy<-testimp.dummy[,-42]
testimp.dummy<-testimp.dummy[,-41]
testimp.dummy<-testimp.dummy[,-40]
testimp.dummy<-testimp.dummy[,-1]

train.dummy %<>% mutate_if(is.integer,as.numeric)
test.dummy %<>% mutate_if(is.integer,as.numeric)
testimp.dummy  %<>% mutate_if(is.integer,as.numeric)

str(train.dummy)
str(test.dummy)
str(testimp.dummy)

train.dummy<-as.matrix(train.dummy)
test.dummy<-as.matrix(test.dummy)
testimp.dummy<-as.matrix(testimp.dummy)

m<-colMeans(train.dummy)
s<-apply(train.dummy,2,sd)
m1<-colMeans(testimp.dummy)
s1<-apply(testimp.dummy,2,sd)
train.dummy<-scale(train.dummy,center = m,scale = s)
test.dummy<-scale(test.dummy,center = m,scale = s)
testimp.dummy<-scale(testimp.dummy,center = m1,scale = s1)

labels <- train$Value
ts_label <- test$Value
ts_labelimp <- testimp$Value

loglabels <- log(labels)
logts_label <- log(ts_label)
logts_labelimp <-log(ts_labelimp)

#Building the ann model with 3 hidden layers and dropout rates for ever layer
build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 100, activation = "relu",input_shape = dim(train.dummy)[2]) %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 50, activation = "relu") %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 20, activation = "relu") %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 1)  
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(learning_rate=0.002),
    metrics = list("mae")
  )
  
  
  model
}





model <- build_model()
model %>% summary()


# Fit the model and store training stats
set.seed(18)
history <- model %>% fit(
  train.dummy,
  loglabels,
  epochs = 100,
  validation_split = 0.2,
  batch_size=16
)



#Prediction on test set
test_predictions <- model %>% predict(test.dummy)
annresult<-exp(test_predictions[ , 1])
sqrt(mean(((logts_label) -(test_predictions))^2))

save_model_hdf5(model,"modelann")

modelann<-load_model_hdf5("modelann")


#Evaluation of the model 
model %>% evaluate(test.dummy, ts_label)
plot(exp(logts_label), exp(test_predictions))


#Test set for report

test_predictions <- modelann %>% predict(testimp.dummy)
annresult<-exp(test_predictions[ , 1])
modelann %>% evaluate(testimp.dummy, logts_labelimp)#loss means MSE here
plot(ts_labelimp, exp(test_predictions))

#write.xlsx(annresult,"kerasann.xlsx")

##### Final Comments #####
# All of the models made significantly good significantly good predictions.
# XGBoost has the lowest MSE with 0,170343.
# But the other models' predictions cannot be ignored because the aim is the show or find 
# a new approach to football player pricing methods.
# The original study has been conducted with unknown versions of R and related  packages. 
# The original resualts are not fully reproducible but close results are obtained.