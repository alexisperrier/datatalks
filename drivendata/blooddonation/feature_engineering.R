'''
This script compares several metrics and engineered feature sets
using Random Forest Feature Importance (MDI and MDA)
The initial data set is the Blood Donation dataset available on UCI
https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
and made available via the driven data competition
http://www.drivendata.org/competitions/2/data/

Author: Alex Perrier alexis.perrier@gmail.com

'''

library('readr')
library('caret')
library('pROC')
library('corrplot')
require(randomForest)

my_logLoss <- function(act, pred){
  # Computes the logLoss between a truth vector and a prediction
  #
  # Args:
  #   act: The truth vector
  #   pred: The prediction. Bothe vectors should have the same length
  #
  # Returns:
  #   the logLoss

  # Error handling
  if (length(act) != length(pred)) {
    stop("Arguments act and pred have different lengths: ",
         length(act), " and ", length(pred), ".")
  }

  eps   <- 1e-15
  nr    <- nrow(pred)
  pred  <- sapply( pred, function(x) max(eps,x))
  pred  <- sapply( pred, function(x) min(1-eps,x))
  ll    <- ( -1 / length(act) ) * sum( act * log(pred) + (1-act) * log(1-pred) )
  return(ll)
}

# Cross Validation and Grid Search wrapper
cv_wrapper <- function(X, features, ctrl){
  # Returns the optimal Random Forest model with regards to
  #   the ctrl cross validation setup (see trainControl)
  #   mtry the level of splits in the forest chosen from the grid search forestGrid
  #
  # Args:
  #   X: The training data frame with X$C as target and features as predictors / columns
  #   features: The predictors
  #   ctrl: the cross validation setup
  #
  # Returns:
  #   the best trained Random Forest according to teh defined metric: logLoss

  set.seed(1967)
  forestGrid <- expand.grid( mtry=seq(1, length(features)-1, by=1) )
  mod <- train(C ~ ., data  = X[ ,features],
              method    = "rf",
              metric    = "logLoss",
              maximize  = FALSE,
              trControl = ctrl,
              tuneGrid  = forestGrid )
  return(mod)
}


# Training and Prediction with Random Forest
training <- function(train, test, features, mtry){
  # Trains a Random Forest model on the train set and predicts the test set.
  #
  # Args:
  #   train: The training set
  #   test: The test set
  #   features: The predictors
  #   mtry: the split level for each tree
  #
  # Returns:
  #   The random forest model, including its predictions for the test set

  set.seed(1967)

  X <- train[ , features[features != "C"]]
  y <- train[ , "C"]
  X_test <- test[ , features[features != "C"]]
  y_test <- test[ , "C"]
  # train RF on train set
  mod <- randomForest(x = X, y = y, xtest = X_test, ytest = y_test,
                      importance=TRUE, mtry = mtry, ntree = 500)
  return(mod)
}

scoring <- function(preds, mod){
  # Given a random forest model, a training and a test set, will print the confusion matrix
  # and compute the following scores: Accuracy, AUC and logLoss
  #
  # Args:
  #   preds: The prediction target with 'Y','N' levels
  #   mod: The random forest model with mod$test$predicted & mod$test$votes vectors
  #
  # Returns:
  #   a vector with Accuracy, AUC and logLoss

  cfm <- confusionMatrix(mod$test$predicted, preds)
  print("Confusion Matrix")
  print(cfm$table)
  print(paste("Accuracy:", format(round(cfm$overall[['Accuracy']], 4), nsmall = 4) ))

  roc <- roc(response = preds, predictor = mod$test$votes[,"Y"], levels = rev(levels(preds)))
  print(paste("AUC:",format(round(roc$auc, 4), nsmall = 4) ))

  ll  <- my_logLoss(ifelse(preds == 'Y', 1, 0 ), mod$test$votes[,"Y"])
  print(paste("logLoss: ", format(round(ll, 4), nsmall = 4) ) )

  return(c(cfm$overall[['Accuracy']], roc$auc, ll ))
}

process <- function(train, test, features){
  # Given a random forest model, a training and a test set, will perform the following sequence of
  # operations:
  #   - Cross Validation and Grid Search
  #   - RandomForest Training
  #   - Scoring
  #
  # Args:
  #   train, test and features
  #
  # Returns:
  #   a vector with mtry, Accuracy, AUC, logLoss and logLoss obtained during Cross Validation

  cv_mod   <- cv_wrapper(train, features, ctrl)
  print(cv_mod$results)
  print(cv_mod$bestTune)

  mod      <- training(train, test, features, cv_mod$bestTune[,"mtry"])

  res      <- scoring(test$C, mod )

  # global variables to remember all features importance across all sets
  imp.gini <<- sort(mod$importance[,'MeanDecreaseGini'], decreasing=TRUE)
  imp.accu <<- sort(mod$importance[,'MeanDecreaseAccuracy'], decreasing=TRUE)

  print(sort(mod$importance[,'MeanDecreaseGini'], decreasing=TRUE)[1:min(5,length(features) -1 ) ])
  print(sort(mod$importance[,'MeanDecreaseAccuracy'], decreasing=TRUE)[1:min(5,length(features) -1 ) ])

  c( cv_mod$bestTune[,"mtry"],
      lapply(
          c(res, cv_mod$results[cv_mod$bestTune[,"mtry"],2] ),
          function(x) format(round(x, 4), nsmall = 4)
      )
  )
}

# initiate
setwd("~/apps/kaggle/ddblood/R")
options(digits=4)
# load data
train <- read_csv('../data/ddblood_train.csv')

# rename columns
colnames(train) <- c("Id","Recency", "Frequency", "Monetary",  "Time", "C")

features.original <- c("Recency", "Frequency", "Time")
# drop ID
train <- train[, c(features.original, "C") ,drop=FALSE]

# convert to numeric
train[, features.original] <- sapply( train[, features.original], as.numeric)

# set target with valid R names
train[,"C"] <- ifelse(train$C == 1, 'Y', 'N' )
train[,"C"] <- factor(train[,"C"])

# create hold-out test set
set.seed(1967)
inTrainingSet <- createDataPartition(train$C, p = .80, list = FALSE)
test   <- train[-inTrainingSet,]
train  <- train[ inTrainingSet,]

# Initialize global variables
results <- data.frame(set= 1:10, mtry = 0, Acc = 0, AUC = 0, logLoss = 0, logLossCV = 0)
# These will allow us to track feature importance accross all feature sets
# Gini (MDI)
imp.gini <- c()
# Accuracy or permutation (MDA)
imp.accu <- c()

# Cross validation setup (10 folds, repeated 10 times, with logLoss for metric)
ctrl <- trainControl(method  = "cv",
                     repeats = 10,
                     number  = 10,
                     summaryFunction = mnLogLoss,
                     classProbs = TRUE)

# Baseline

features.baseline <- c("C", features.original)
res         <- process(train, test, features.baseline)
results[1,] <- c('Baseline', res)
print(results)

# Set 1: Log, SQRT and SQ
print("== Set 1: Log, SQRT and SQ")
# Log of original features
train$logRecency    <- log(train$Recency + 1)
train$logFrequency  <- log(train$Frequency + 1)
train$logTime       <- log(train$Time + 1)

# Square root of original features
train$sqrtRecency   <- sqrt(train$Recency)
train$sqrtFrequency <- sqrt(train$Frequency)
train$sqrtTime      <- sqrt(train$Time)

# Square of original features
train$sqRecency     <- train$Recency **2
train$sqFrequency   <- train$Frequency **2
train$sqTime        <- train$Time **2

# and the same for the test set: log, square root and square
test$logRecency    <- log(test$Recency + 1)
test$logFrequency  <- log(test$Frequency + 1)
test$logTime       <- log(test$Time + 1)

test$sqrtRecency   <- sqrt(test$Recency)
test$sqrtFrequency <- sqrt(test$Frequency)
test$sqrtTime      <- sqrt(test$Time)

test$sqRecency     <- test$Recency **2
test$sqFrequency   <- test$Frequency **2
test$sqTime        <- test$Time **2

# define Set 1 features
features.set_1 <- c("logRecency", "logFrequency", "logTime",
                    "sqrtRecency", "sqrtFrequency", "sqrtTime",
                    "sqRecency", "sqFrequency", "sqTime")
res         <- process(train, test, c(features.baseline, features.set_1))
results[2,] <- c('Set 1', res)
print(results)

print("== Set 2: Cross features: products and ratios")

# Calculate some multiples between the original features
train$RecencyFrequency      <- train$Recency * train$Frequency
train$FrequencyTime         <- train$Frequency * train$Time
train$RecencyTime           <- train$Recency * train$Time
train$RecencyFrequencyTime  <- train$Recency * train$Frequency * train$Time

# and some ratios
train$ratioRecencyFrequency <- train$Recency / (train$Frequency + 1)
train$ratioFrequencyTime    <- train$Frequency / (train$Time + 1)
train$ratioRecencyTime      <- train$Recency / (train$Time + 1)

# same multiples for the test set
test$RecencyFrequency      <- test$Recency * test$Frequency
test$FrequencyTime         <- test$Frequency * test$Time
test$RecencyTime           <- test$Recency * test$Time
test$RecencyFrequencyTime  <- test$Recency * test$Frequency * test$Time

# and same ratios
test$ratioRecencyFrequency <- test$Recency / (test$Frequency + 1)
test$ratioFrequencyTime    <- test$Frequency / (test$Time + 1)
test$ratioRecencyTime      <- test$Recency / (test$Time + 1)

# define Set 2 features
features.set_2 <- c("ratioRecencyFrequency", "ratioFrequencyTime", "ratioRecencyTime",
                    "RecencyFrequency", "FrequencyTime", "RecencyTime", "RecencyFrequencyTime")
res         <- process(train, test, c(features.baseline, features.set_2))
results[3,] <- c('Set 2', res)

print(results)

print("== Set 3: Feature interpretation")

# People who have not given after their first time (or month)
# while keeping track of the number of times they gave (train$Frequency)
train$OnlyOnce <- ifelse((train$Recency == train$Time) , train$Frequency, 0)
test$OnlyOnce  <- ifelse((test$Recency == test$Time) , test$Frequency, 0)

# Regular donors have given at least once every N month for longer than 6 months
train$Regular <- 0
for (n in 10:2) {
    idx <- ( train$Frequency > (train$Time - train$Recency) /n ) & (train$Time - train$Recency) > 6
    train[,"Regular"] <- ifelse(test = idx,yes = n,train[,"Regular"])
}

test$Regular <- 0
for (n in 10:2) {
    idx <- ( test$Frequency >  (test$Time - test$Recency) /n  )  & (test$Time - test$Recency) > 6
    test[,"Regular"] <- ifelse(test = idx,yes = n,test[,"Regular"])
}

# define Set 3 features
features.set_3 <- c("OnlyOnce", "Regular")
res         <- process(train, test, c(features.baseline, features.set_3) )
results[4,] <- c('Set 3', res )

print(results)

# combining set 2 and set 3
res         <- process(train, test, c(features.baseline, features.set_2, features.set_3) )
results[5,] <- c('Set 2+3', res )
print(results)

# combining set 1 and set 3
res         <- process(train, test, c(features.baseline, features.set_1, features.set_3) )
results[6,] <- c('Set 1+3', res  )
print(results)

# combining set 1 and set 2
res         <- process(train, test, c(features.baseline, features.set_1, features.set_2) )
results[7,] <- c('Set 1+2', res  )
print(results)

# combining set 1, 2 and set 3
res         <- process(train, test, c(features.baseline, features.set_1, features.set_2, features.set_3) )
results[8,] <- c('Set 1+2+3', res  )
print(results)

# Score for most important features per set
# mig stands for Most Important Gini
# mia stands for Most Important Accuracy

# select most important features for Gini
mig =  rep(0, length(c(unique(names(imp.gini)))))
names(mig) = c(unique(names(imp.gini)))

for (i in 1:length(imp.gini)) {
    n <- names(imp.gini[i])
    mig[ n ] <- mig[ n ] + imp.gini[[i]]
}
print(sort(mig, decreasing=TRUE))

# select most important features for Accuracy
mia =  rep(0, length(c(unique(names(imp.accu)))))
names(mia) = c(unique(names(imp.accu)))

for (i in 1:length(imp.accu)) {
    n <- names(imp.accu[i])
    mia[ n ] <- mia[ n ] + imp.accu[[i]]
}
print(sort(mia, decreasing=TRUE))

# Set 9: most important features (Gini)
# We keep the features whose aggregated score is above the median score of all feature importances
features    <- c("C", names(mig[mig > median(mig)]))
res         <- process(train, test, features)
results[9,] <- c('Imp Gini', res )

print(results)

# Set 10: most important features (Accuracy)
features     <- c("C", names(mia[mia > median(mia)]))
res          <- process(train, test, features)
results[10,] <- c('Imp Accuracy', res )

print(results)



