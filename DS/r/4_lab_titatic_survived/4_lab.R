setwd('~/study/r/DS/')
library(caret)

library(caret)
library(dplyr)

# Загрузка данных
titanic <- read.csv('titanic3.csv')

head(titanic)

# Очистка данных
titanic <- titanic %>%
  select(survived, pclass, sex, age, sibsp, parch, fare, embarked) %>%
  mutate(sex = as.factor(sex),
         embarked = as.factor(replace(embarked, is.na(embarked), "S")),
         pclass = as.factor(pclass))
head(titanic)

titanic$age[is.na(titanic$age)] <- median(titanic$age, na.rm = TRUE)

# Разделение выборки
set.seed(1021)
train_index <- createDataPartition(titanic$survived, p = 0.7, list = FALSE)
train_data <- titanic[train_index, ]
test_data <- titanic[-train_index, ]

# Список всех переменных (кроме intercept)
vars <- c("age", "pclass", "sex", "sibsp", "parch", "fare")

# Инициализация переменных для лучшей модели
best_aic <- Inf
final_model <- NULL
best_formula <- NULL

# Перебор возможных комбинаций переменных от максимального количества к минимальному

for (i in length(vars):1) {
  combinations <- combn(vars, i, simplify = FALSE)
  isDone <- TRUE
  
  for (combo in combinations) {
    # Строим модель для текущей комбинации переменных
    formula <- as.formula(paste("survived ~", paste(combo, collapse = " + ")))
    model <- glm(formula, data = train_data, family = binomial)
    
    # Оцениваем AIC для этой модели
    model_aic <- AIC(model)
    
    # Если AIC лучше, обновляем лучшие параметры
    if (model_aic < best_aic) {
      best_aic <- model_aic
      final_model <- model
      best_formula <- formula
      isDone <- FALSE
    }
  }
  # Данная проверка служит для предотвращения лишней работы по проверке модели 
  # на меньшем количестве параметров. 
  # В случае, если проверка на не даёт улучшения
  # AIC, то последняя лучшая модель является оптимальной 
  if (isDone){
    break
  }
}

# Оценка лучшей модели на тестовых данных
predicted_survival <- predict(final_model, newdata = test_data, type = "response")
predicted_class <- ifelse(predicted_survival > 0.5, 1, 0)

conf_matrix <- confusionMatrix(factor(test_data$survived, levels = c(0, 1)), 
                     factor(predicted_class, levels = c(0, 1)))


# Вывод результатов
cat("Финальная модель:", deparse(best_formula), "\n")
cat("AIC модели:", AIC(final_model), "\n")

summary(final_model)
conf_matrix
coefs <- coefficients(final_model)
coefs

# Функция для предсказания вероятности выживания для датафрейма
predict_survival_df <- function(data, coefs) {
  # Кодируем pclass в dummy-переменные
  data$pclass2nd <- ifelse(data$pclass == "2nd", 1, 0)
  data$pclass3rd <- ifelse(data$pclass == "3rd", 1, 0)
  
  # Кодируем пол (1 - мужчина, 0 - женщина)
  data$sexmale <- ifelse(data$sex == "male", 1, 0)
  
  # Вычисляем логиты
  logit_p <- coefs["(Intercept)"] + 
    coefs["age"] * data$age +
    coefs["pclass2nd"] * data$pclass2nd +
    coefs["pclass3rd"] * data$pclass3rd +
    coefs["sexmale"] * data$sexmale +
    coefs["sibsp"] * data$sibsp
  
  # Преобразуем логиты в вероятность
  data$survival_probability <- 1 / (1 + exp(-logit_p))
  
  return(data)
}

# Пример датафрейма
passengers <- data.frame(
  age = c(30, 22, 40),
  pclass = c("3rd", "1st", "2nd"),
  sex = c("male", "female", "male"),
  sibsp = c(1, 0, 2)
)
passengers$pclass <- factor(passengers$pclass, levels = c("1st", "2nd", "3rd"))


# Применяем формулу
predicted_passengers <- predict_survival_df(passengers, coefs)

# Вывод результата формулы
print(predicted_passengers[, c("age", "pclass", "sex", "sibsp", "survival_probability")])

# Применяем предсказание модели
model_predicted_survival <- predict(final_model, newdata = passengers, type = "response")

# Склеиваем данные и выводим в требуемом формате
print(cbind(passengers, survival_probability = model_predicted_survival))


# Добавляем предсказания в test_data
test_data$predicted_probability <- predicted_survival
test_data$predicted_class <- predicted_class

# 1. False Positives (FP) - предсказано "выжил", но реально "не выжил" (survived = 0)
false_positives <- test_data[test_data$predicted_class == 1 & test_data$survived == 0, ]

# 2. False Negatives (FN) - предсказано "не выжил", но реально "выжил" (survived = 1)
false_negatives <- test_data[test_data$predicted_class == 0 & test_data$survived == 1, ]

# Вывод результатов
cat("False Positives (ошибочные предсказания 'выживания'):\n")
summary(false_positives)

cat("\nFalse Negatives (ошибочные предсказания 'смерти'):\n")
summary(false_negatives)

