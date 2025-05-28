library(caret)

set.seed(2004)

train_index <- caret::createDataPartition(dd$Drinks_per_week, p = 0.7, list = FALSE)
train_data <- dd[train_index, ]
test_data <- dd[-train_index, ]

dim(train_data)
dim(test_data)
