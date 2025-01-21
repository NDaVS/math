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
summary(data$class)

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
std_errors <- coef_summary$standard.errors

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

z_scores <- coefficients / std_errors
p_values <- 2 * (1 - pnorm(abs(z_scores)))

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
validation_pred_prob <- predict(model, test_data, type="prob")
head(validation_predictions)
head(validation_pred_prob)


misclassification=table(validation_predictions, test_data$class)
misclassification

correct_predictions <- sum(diag(misclassification))

# Общее количество предсказаний
total_predictions <- sum(misclassification)

# Вычисление точности
accuracy <- correct_predictions / total_predictions

# Вывод точности
print(paste("Точность модели:", round(accuracy * 100, 2), "%"))

# Убедитесь, что установлены необходимые пакеты
if (!require("pROC")) install.packages("pROC")
library(pROC)

# Предсказание вероятностей для всех классов
validation_pred_prob <- predict(model, validation_data, type = "prob")

# Инициализация для хранения результатов
roc_list <- list()
auc_list <- c()

# One-vs-Rest для каждого класса
for (class in colnames(validation_pred_prob)) {
  # Бинаризуем метки: текущий класс = 1, остальные = 0
  binary_true_labels <- ifelse(validation_data$class == class, 1, 0)
  
  # Построение ROC-кривой
  roc_curve <- roc(binary_true_labels, validation_pred_prob[, class], quiet = TRUE)
  roc_list[[class]] <- roc_curve
  
  # Сохраняем AUC
  auc_list <- c(auc_list, auc(roc_curve))
  
  # Выводим AUC для текущего класса
  cat(paste("Класс:", class, "- AUC:", round(auc(roc_curve), 3)), "\n")
}

# Визуализация ROC-кривых
plot(roc_list[[1]], col = "blue", main = "ROC-кривые (One-vs-Rest)", lwd = 2)
for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = i, add = TRUE, lwd = 2)
}
legend("bottomright", legend = colnames(validation_pred_prob), col = 1:length(roc_list), lwd = 2)

# Гистограмма распределения по количеству комнат (nrooms) для тестового набора
ggplot(test_data, aes(x = nrooms, fill = class)) +
  geom_histogram(binwidth = 1, position = "dodge", color = "black") +
  labs(
    title = "Распределение квартир по количеству комнат (nrooms) для тестового набора",
    x = "Количество комнат (nrooms)",
    y = "Количество квартир"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")

# Гистограмма распределения по количеству комнат (nrooms) для предсказанных значений
predicted_test_data <- test_data
predicted_test_data$class <- validation_predictions  # Добавляем предсказанные классы

ggplot(predicted_test_data, aes(x = nrooms, fill = class)) +
  geom_histogram(binwidth = 1, position = "dodge", color = "black") +
  labs(
    title = "Распределение предсказанных значений по количеству комнат (nrooms)",
    x = "Количество комнат (nrooms)",
    y = "Количество квартир"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")

summary(test_data$class)
summary(predicted_test_data$class)



model <- ksvm(class ~ ., data = train_data, kernel = "rbfdot", prob.model = TRUE)

# Обзор модели
print(model)

# Оценка модели на валидационных данных
validation_predictions <- predict(model, validation_data, type = "response")
validation_pred_prob <- predict(model, validation_data, type = "probabilities")

# Метрики классификации
misclassification <- table(validation_predictions, validation_data$class)
print(misclassification)

accuracy <- sum(diag(misclassification)) / sum(misclassification)
cat("\nТочность модели на валидационных данных:", round(accuracy * 100, 2), "%\n")

# Оценка модели на тестовых данных
test_predictions <- predict(model, test_data, type = "response")
test_pred_prob <- predict(model, test_data, type = "probabilities")

# Метрики классификации на тестовых данных
misclassification_test <- table(test_predictions, test_data$class)
print(misclassification_test)

accuracy_test <- sum(diag(misclassification_test)) / sum(misclassification_test)
cat("\nТочность модели на тестовых данных:", round(accuracy_test * 100, 2), "%\n")

# ROC-анализ для SVM
if (!require("pROC")) install.packages("pROC")
library(pROC)

roc_list <- list()
auc_list <- c()

# One-vs-Rest для каждого класса
for (class in colnames(validation_pred_prob)) {
  binary_true_labels <- ifelse(validation_data$class == class, 1, 0)
  roc_curve <- roc(binary_true_labels, validation_pred_prob[, class], quiet = TRUE)
  roc_list[[class]] <- roc_curve
  auc_list <- c(auc_list, auc(roc_curve))
  cat(paste("Класс:", class, "- AUC:", round(auc(roc_curve), 3)), "\n")
}

# Визуализация ROC-кривых
plot(roc_list[[1]], col = "blue", main = "ROC-кривые (One-vs-Rest)", lwd = 2)
for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = i, add = TRUE, lwd = 2)
}
legend("bottomright", legend = colnames(validation_pred_prob), col = 1:length(roc_list), lwd = 2)

# Визуализация результатов
ggplot(test_data, aes(x = nrooms, fill = class)) +
  geom_histogram(binwidth = 1, position = "dodge", color = "black") +
  labs(
    title = "Распределение квартир по количеству комнат (nrooms) для тестового набора",
    x = "Количество комнат (nrooms)",
    y = "Количество квартир"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")

predicted_test_data <- test_data
predicted_test_data$class <- test_predictions

ggplot(predicted_test_data, aes(x = nrooms, fill = class)) +
  geom_histogram(binwidth = 1, position = "dodge", color = "black") +
  labs(
    title = "Распределение предсказанных значений по количеству комнат (nrooms)",
    x = "Количество комнат (nrooms)",
    y = "Количество квартир"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")