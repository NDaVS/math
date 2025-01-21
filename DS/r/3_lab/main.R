setwd("D:\\Coding\\math\\DS\\r\\3_lab")
data <- read.table(file="moscow.txt", header = TRUE)
summary(data)
dim(data)
print(colSums(sapply(data, is.na)))

set.seed(42)


library(nnet)
library(ggplot2)
library(dplyr)
library(caret)


data$class <- cut(data$totsp, breaks = 4, labels = c("small", "medium", "large", "huge"))

data <- select(data, -totsp)
dim(data)
summary(data)

train_index <- caret::createDataPartition(data$class, p = 0.6, list = FALSE)
train_data <- data[train_index, ]
temp_data <- data[-train_index, ]

validation_index <- createDataPartition(temp_data$class, p = 0.5, list = FALSE)
validation_data <- temp_data[validation_index, ]
test_data <- temp_data[-validation_index, ]

cat("Size of train data:", nrow(train_data), "\n")
cat("Size of validation data:", nrow(validation_data), "\n")
cat("Size of test data:", nrow(test_data), "\n")

cat("Size of train data:", dim(train_data), "\n")
cat("Size of validation data:", dim(validation_data), "\n")
cat("Size of test data:", dim(test_data), "\n")



model <- multinom(class ~ ., data = train_data)

summary(model)

# Оценка модели на валидационной выборке
validation_predictions <- predict(model, validation_data)
validation_pred_prob <- predict(model, validation_data, type="prob")
head(validation_predictions)
head(validation_pred_prob)


misclassification=table(validation_predictions, validation_data$class)
misclassification

correct_predictions <- sum(diag(misclassification))

# Общее количество предсказаний
total_predictions <- sum(misclassification)

# Вычисление точности
accuracy <- correct_predictions / total_predictions

# Вывод точности
print(paste("Точность модели:", round(accuracy * 100, 2), "%"))


coef_summary <- summary(model)

# Извлечение коэффициентов
coefficients <- coef_summary$coefficients
print(coefficients)
# Добавление имен колонок

# Теперь извлекаем коэффициенты и стандартные ошибки
estimates <- coefficients
std_errors <- std_errors

# Уровень доверия
alpha <- 0.05
z_alpha_over_2 <- qnorm(1 - alpha / 2)  # Критическое значение z

# Вычисление доверительных интервалов
lower_bounds <- estimates - z_alpha_over_2 * std_errors
upper_bounds <- estimates + z_alpha_over_2 * std_errors

# Создание таблицы доверительных интервалов
confidence_intervals <- data.frame(
  Coefficient = estimates,
  Lower_Bound = lower_bounds,
  Upper_Bound = upper_bounds
)

# Вывод доверительных интервалов
print(confidence_intervals)

cat("\nP-value:\n")
print(p_values)

# AIC
AIC(model)


model <- multinom(class ~ nrooms + livesp, data = train_data)

summary(model)

# Оценка модели на валидационной выборке
validation_predictions <- predict(model, validation_data)
validation_pred_prob <- predict(model, validation_data, type="prob")
head(validation_predictions)
head(validation_pred_prob)


misclassification=table(validation_predictions, validation_data$class)
misclassification

correct_predictions <- sum(diag(misclassification))

# Общее количество предсказаний
total_predictions <- sum(misclassification)

# Вычисление точности
accuracy <- correct_predictions / total_predictions

# Вывод точности
print(paste("Точность модели:", round(accuracy * 100, 2), "%"))

coef_summary <- summary(model)
coefficients <- coef_summary$coefficients  # Коэффициенты модели
std_errors <- coef_summary$standard.errors  # Стандартные ошибки

# Вычисление z-статистик и p-значений
z_scores <- coefficients / std_errors
p_values <- 2 * (1 - pnorm(abs(z_scores)))


cat("\nP-value:\n")
print(p_values)

coef_summary <- summary(model)

# Извлечение коэффициентов
coefficients <- coef_summary$coefficients
print(coefficients)
# Добавление имен колонок

# Теперь извлекаем коэффициенты и стандартные ошибки
estimates <- coefficients
std_errors <- std_errors

# Уровень доверия
alpha <- 0.05
z_alpha_over_2 <- qnorm(1 - alpha / 2)  # Критическое значение z

# Вычисление доверительных интервалов
lower_bounds <- estimates - z_alpha_over_2 * std_errors
upper_bounds <- estimates + z_alpha_over_2 * std_errors

# Создание таблицы доверительных интервалов
confidence_intervals <- data.frame(
  Coefficient = estimates,
  Lower_Bound = lower_bounds,
  Upper_Bound = upper_bounds
)

# Вывод доверительных интервалов
print(confidence_intervals)
# AIC
AIC(model)



validation_predictions <- predict(model, test_data)
validation_pred_prob <- predict(model, validation_data, type="prob")
head(validation_predictions)
head(validation_pred_prob)


misclassification=table(validation_predictions, validation_data$class)
misclassification

correct_predictions <- sum(diag(misclassification))

# Общее количество предсказаний
total_predictions <- sum(misclassification)

# Вычисление точности
accuracy <- correct_predictions / total_predictions

# Вывод точности
print(paste("Точность модели:", round(accuracy * 100, 2), "%"))

