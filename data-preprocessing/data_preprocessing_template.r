dataset = read.csv('Data_Preprocessing/Data.csv')

dataset$Age = ifelse(is.na(dataset$Age),
    ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
    dataset$Age
)

dataset$Salary = ifelse(is.na(dataset$Salary),
    ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
    dataset$Salary
)

# Encoding categorical data
dataset$Country = factor(dataset$Country,
    levels = c('France', 'Germany', 'Spain'),
    labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
    levels = c('Yes', 'No'),
    labels = c(0, 1))

# Split into testing and training sets
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio=0.8)

training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Feature scaling
training_set[, 2:3] = scale(training_set[, 2:3])
testing_set[, 2:3] = scale(testing_set[, 2:3])
