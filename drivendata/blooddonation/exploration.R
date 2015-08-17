# This script explores the DrivenData Blood Donation competition data set
# The data is available at http://www.drivendata.org/competitions/2/data/

# Loading the data

library('readr')
library('corrplot')
setwd("your working directory")
train <- read_csv('../data/ddblood_train.csv')
test  <- read_csv('../data/ddblood_test.csv')


# A first look at the train set:
print(str(train))

# Renaming columns
colnames(train) <- c("Id","Recency", "Frequency", "Monetary",  "Time", "C")
colnames(test)  <- c("Id","Recency", "Frequency", "Monetary", "Time")

# Splitting the data
y     <- train$C
train <- train[,c("Recency", "Frequency", "Time") ,drop=FALSE]
test  <- test[ ,c("Recency", "Frequency", "Time") ,drop=FALSE]
print(str(train))


# Train / Test comparison

# Figure 1 shows boxplots of the features for the train and test sets.
# We can see that the train and test sets have very similar distributions.
# Some outliers (Recency > 30 or Frequency > 20) are clearly visible.

par(bty = 'n')
boxplot(train,
        boxwex = 0.28, at = 1:3 - 0.2,
        col = "light blue",
        main = "Blood Donation",
        xlab = "", ylab = "", cex.axis=0.85, bty="n", xaxt="n", yaxt="n")
boxplot(test, add = TRUE,
        boxwex = 0.28, at = 1:3 + 0.2,
        col = "bisque", cex.axis=0.85, xaxt="n")
legend(1.5, 80, c("Train", "Test"),
       fill = c("light blue", "bisque"))
axis(1,side = 1, at=c(1,2,3),labels=c('Recency','Frequency','Time'),tick=FALSE)


# Check the outliers fo train$Recency
index <- train$Recency > 30
print(train[index, ])
print(y[index])

# Check the outliers for train$Frequency
index <- train$Frequency > 30
print(train[index, ])
print(y[index])

# Donators vs non Donators comparison

donors <- y == 1

par(bty = 'n')
boxplot(train[donors,],
        boxwex = 0.18, at = 1:3 - 0.2,
        col = "light blue",
        main = "Blood Donation - March Donation vs March No Donation",
        xlab = "", ylab = "", cex.axis=0.85, bty="n", xaxt="n", yaxt="n")
boxplot(train[!donors,], add = TRUE,
        boxwex = 0.38, at = 1:3 + 0.2,
        col = "bisque", cex.axis=0.85, xaxt="n")
legend(1, 95, c("Donation in March", "No Donation in March"),
       fill = c("light blue", "bisque"))
axis(1,side = 1, at=c(1,2,3),labels=c('Recency','Frequency','Time'),tick=FALSE)


# Correlation
corrplot.mixed(cor(train), lower = "ellipse", upper="number")

# Multiple donations in the same month?
# more than once
index <- (train$Recency == train$Time) & train$Frequency > 1
print(dim(train[index,]))  # 26, 3

# more than twice
index <- (train$Recency == train$Time) & train$Frequency > 2
print(dim(train[index,]))  # 3, 3

