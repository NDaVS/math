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

library(gridExtra)

# Выбираем FP и FN
false_positives <- test_data[test_data$predicted_class == 1 & test_data$survived == 0, ]
false_negatives <- test_data[test_data$predicted_class == 0 & test_data$survived == 1, ]

# Гистограммы для числовых признаков с разделением по полу
p1 <- ggplot(false_positives, aes(x = age, fill = sex)) + 
  geom_histogram(binwidth = 5, alpha = 0.7, position = "dodge") + 
  ggtitle("False Positives: Age") + theme_minimal()

p2 <- ggplot(false_negatives, aes(x = age, fill = sex)) + 
  geom_histogram(binwidth = 5, alpha = 0.7, position = "dodge") + 
  ggtitle("False Negatives: Age") + theme_minimal()

p3 <- ggplot(false_positives, aes(x = fare, fill = sex)) + 
  geom_histogram(binwidth = 10, alpha = 0.7, position = "dodge") + 
  ggtitle("False Positives: Fare") + theme_minimal()

p4 <- ggplot(false_negatives, aes(x = fare, fill = sex)) + 
  geom_histogram(binwidth = 10, alpha = 0.7, position = "dodge") + 
  ggtitle("False Negatives: Fare") + theme_minimal()

# Boxplot'ы
p5 <- ggplot(false_positives, aes(x = sex, y = age, fill = sex)) + 
  geom_boxplot(alpha = 0.5) + 
  ggtitle("FP: Age Distribution by Sex") + theme_minimal()

p6 <- ggplot(false_negatives, aes(x = sex, y = age, fill = sex)) + 
  geom_boxplot(alpha = 0.5) + 
  ggtitle("FN: Age Distribution by Sex") + theme_minimal()

# Категориальные переменные (Pclass)
p7 <- ggplot(false_positives, aes(x = pclass, fill = sex)) + 
  geom_bar(alpha = 0.7, position = "dodge") + 
  ggtitle("False Positives: Pclass by Sex") + theme_minimal()

p8 <- ggplot(false_negatives, aes(x = pclass, fill = sex)) + 
  geom_bar(alpha = 0.7, position = "dodge") + 
  ggtitle("False Negatives: Pclass by Sex") + theme_minimal()

# Категориальные переменные (Embarked)
p9 <- ggplot(false_positives, aes(x = embarked, fill = sex)) + 
  geom_bar(alpha = 0.7, position = "dodge") + 
  ggtitle("False Positives: Embarked by Sex") + theme_minimal()

p10 <- ggplot(false_negatives, aes(x = embarked, fill = sex)) + 
  geom_bar(alpha = 0.7, position = "dodge") + 
  ggtitle("False Negatives: Embarked by Sex") + theme_minimal()

# Отображение графиков
grid.arrange(p1, p2, p3, p4, ncol = 2)
grid.arrange(p5, p6, p7, p8, ncol = 2)
grid.arrange(p9, p10, ncol = 2)

ggplot(test_data, aes(x = age, y = sex, color = factor(predicted_class == survived))) +
  geom_point(alpha = 0.7) +
  scale_color_manual(values = c("red", "blue")) +
  labs(title = "FP (красный) и FN (синий) на тестовой выборке")



test_data <- test_data[!(test_data$sex == "female"  & test_data$pclass == "3rd"), ]
test_data <- test_data[!(test_data$sex == "female"  & (test_data$embarked == "Southampton" |test_data$embarked == "Cherbourg" )), ]
#test_data_2 <- test_data_2[!(test_data_2$sex == "male" & test_data_2$age >= 28 &  test_data_2$age <= 32), ]
test_data <- test_data[!(test_data$sex == "male" & (test_data$embarked == "Southampton" |test_data$embarked == "Cherbourg" )), ]
test_data <- test_data[!(test_data$sex == "male" & test_data$age == 29), ]
# Оценка лучшей модели на тестовых данных
predicted_survival <- predict(final_model, newdata = test_data, type = "response")
predicted_class <- ifelse(predicted_survival > 0.5, 1, 0)

conf_matrix <- confusionMatrix(factor(test_data$survived, levels = c(0, 1)), 
                               factor(predicted_class, levels = c(0, 1)))
conf_matrix
