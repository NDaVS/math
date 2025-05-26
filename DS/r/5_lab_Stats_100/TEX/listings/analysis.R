library(caret)
dd$Home_Town[dd$Home_Town < 2] <- 0
dd$Home_Town[dd$Home_Town >= 2] <-1
dd$Gender <- factor(dd$Gender)

train_index <- caret::createDataPartition(dd$Drinks_per_week, p = 0.6, list = FALSE)
train_data <- dd[train_index, ]
temp_data <- dd[-train_index, ]

validation_index <- createDataPartition(temp_data$Drinks_per_week, p = 0.5, list = FALSE)
validation_data <- temp_data[validation_index, ]
test_data <- temp_data[-validation_index, ]
