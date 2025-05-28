#=============== 
setwd('/home/ndavs/study/math/DS/r/5_lab_Stats_100')
# считаем датасет
data <- read.table('data.dat', sep=' ')
dim(data)
head(data)

# сформируем вектор с заголовками
base_headers <- readLines('headers.txt', n=1)
cleaned_header <- gsub("\\s*\\([^()]*\\)", "", base_headers)

cleaned_header <- gsub("\\.", "", cleaned_header)
cleaned_header <- gsub("\\s+", " ", cleaned_header)
cleaned_header <- trimws(cleaned_header)
cleaned_header <- unlist(strsplit(cleaned_header, " "))
length(cleaned_header)

# вставим заголовки в датафрейм
colnames(data) <- cleaned_header
dim(data)
head(data)

# выберем только нужные показатели
dd <- data[, c("Drinks_per_week", "Party_Hours_per_week", "Gender", "Home_Town")]
head(dd)
dim(dd)
summary(dd)

dd$Home_Town[dd$Home_Town < 2] <- 0
dd$Home_Town[dd$Home_Town >= 2] <-1
dd$Gender <- factor(dd$Gender)

library(caret)
set.seed(2004)
train_index <- caret::createDataPartition(dd$Drinks_per_week, p = 0.7, list = FALSE)
train_data <- dd[train_index, ]
test_data <- dd[-train_index, ]

dim(train_data)
dim(test_data)


#========================== 
# сформируем модели линейной регрессии
m1 <- lm("Drinks_per_week ~ Party_Hours_per_week + Gender", data=train_data)
summary(m1)

m2 <-  lm("Drinks_per_week ~ Party_Hours_per_week + Gender + Home_Town", data=train_data)
summary(m2)

#===================
library(ggplot2)

# 1. Диапазон значений по оси X
x_seq <- seq(
  from = min(train_data$Party_Hours_per_week),
  to = max(train_data$Party_Hours_per_week),
  length.out = 100
)

# 2. Предсказания модели m1
newdata_m1 <- expand.grid(
  Party_Hours_per_week = x_seq,
  Gender = levels(train_data$Gender)
)
newdata_m1$Home_Town <- NA  # Добавляем для совместимости
newdata_m1$Pred <- predict(m1, newdata_m1)
newdata_m1$Model <- "m1"
newdata_m1$Group <- newdata_m1$Gender

# 3. Предсказания модели m2
newdata_m2 <- expand.grid(
  Party_Hours_per_week = x_seq,
  Gender = levels(train_data$Gender),
  Home_Town = c(0, 1)
)
newdata_m2$Pred <- predict(m2, newdata_m2)
newdata_m2$Model <- "m2"
newdata_m2$Group <- interaction(newdata_m2$Gender, newdata_m2$Home_Town, sep = "_HT")

# 4. Объединяем предсказания
pred_all <- rbind(newdata_m1, newdata_m2)
train_data$Home_Town <- factor(train_data$Home_Town)
# 5. Общий график
ggplot(train_data, aes(x = Party_Hours_per_week, y = Drinks_per_week)) +
  geom_point(
    aes(color = Gender, shape = Home_Town),
    alpha = 1
  ) +
  geom_line(
    data = pred_all,
    aes(
      x = Party_Hours_per_week,
      y = Pred,
      color = Group,
      linetype = Model
    ),
    size = 1
  ) +
  scale_linetype_manual(
    values = c(m1 = "solid", m2 = "dashed"),
    name = "Модель"
  ) +
  labs(
    title = "Объединённый график моделей m1 и m2",
    subtitle = "Цвет — пол, форма — родной город",
    x = "Party Hours per Week",
    y = "Drinks per Week",
    color = "Пол (Gender)",
    shape = "Родной город (Home_Town)"
  ) +
  theme_minimal()


#=====================
predictions <- predict(m2, newdata = test_data)

residuals <- test_data$Party_Hours_per_week - predictions

SST <- sum((test_data$Party_Hours_per_week - mean(test_data$Party_Hours_per_week))^2)

SSE <- sum(residuals^2)

k <- length(coef(m1)) - 1
n <- nrow(test_data)
SSR <- SST - SSE

F_statistic <- (SSR / k) / (SSE / (n - k - 1))

F_statistic
#=====================

x_max <- max(dd$Party_Hours_per_week)
x_ext <- x_max * c(1.05, 1.10, 1.15)
new_rows <- expand.grid(
  Party_Hours_per_week = x_ext,
  Gender = levels(dd$Gender)
)
new_rows <- expand.grid(Party_Hours_per_week = x_ext, Gender = levels(dd$Gender))
# другие переменные заполним NA или по аналогии
new_rows$Drinks_per_week <- predict(m1, newdata = new_rows)
new_rows$Home_Town <- NA 

# Доверительные интервалы
conf <- predict(m1, newdata = new_rows, interval = "confidence")
# Предиктивные интервалы
pred <- predict(m1, newdata = new_rows, interval = "prediction")
# Объединение с исходными значениями
results <- cbind(new_rows, conf, pred_Lwr = pred[ , "lwr"], pred_Upr = pred[ , "upr"])

print(results)
#===================
# 1. Диапазон оси X с расширением
range_x <- range(test_data$Party_Hours_per_week)
x_min <- range_x[1]
x_max <- range_x[2]
x_range <- x_max - x_min

x_ext <- seq(
  from = x_min,
  to   = x_max + 0.15 * x_range,
  length.out = 200
)

# 2. Генерация новых данных для прогноза по уровням Gender
newdata <- expand.grid(
  Party_Hours_per_week = x_ext,
  Gender = levels(test_data$Gender)
)

# 3. Прогноз: доверительный интервал
conf_pred <- predict(m1, newdata = newdata, interval = "confidence", level = 0.95)
conf_df <- cbind(newdata, conf_pred)
conf_df$Type <- "Confidence"

# 4. Прогноз: предиктивный интервал
pred_pred <- predict(m1, newdata = newdata, interval = "prediction", level = 0.95)
pred_df <- cbind(newdata, pred_pred)
pred_df$Type <- "Prediction"



# 5. Объединяем всё
all_preds <- rbind(conf_df, pred_df)

# 6. График
ggplot() +
  geom_point(data = test_data, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender), alpha = 1) +
  geom_point(data = new_rows, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender, ), alpha = 1) +
  
  # Линия предсказания
  geom_line(data = all_preds, aes(x = Party_Hours_per_week, y = fit, color = Gender), size = 1) +
  
  # Ленты интервалов
  geom_ribbon(
    data = all_preds,
    aes(x = Party_Hours_per_week, ymin = lwr, ymax = upr, fill = Gender),
    alpha = 0.2
  ) +
  
  facet_wrap(~Type) +  # Разделение графика: confidence vs prediction
  
  labs(
    title = "Доверительный и предиктивный интервалы модели m1",
    subtitle = "Прогноз с расширением области на 15% по Party_Hours_per_week",
    x = "Party Hours per Week",
    y = "Drinks per Week"
  ) +
  theme_minimal()

#===================================

#===================================

#Отрисуем модели
library(ggplot2)

ggplot(dd, aes(y = Drinks_per_week, x = Party_Hours_per_week, color = Gender)) +
  geom_point() +  
  scale_color_manual(values = c("blue", "red", "green"), labels = c("male", "female", "other"))+
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  labs(title = "Модель 1: Party Hours per Week vs. Drinks per Week",
       x = "Drinks per Week",
       y = "Party Hours per Week") +
  theme_minimal()

# Получение предсказанных значений и интервалов для модели m1
pred_m1 <- predict(m1, newdata = dd, interval = "confidence")
pred_m1_pred <- predict(m1, newdata = dd, interval = "prediction")

# Добавление предсказанных значений и интервалов в датафрейм
dd$m1_fit <- pred_m1[, "fit"]
dd$m1_lwr_conf <- pred_m1[, "lwr"]
dd$m1_upr_conf <- pred_m1[, "upr"]
dd$m1_lwr_pred <- pred_m1_pred[, "lwr"]
dd$m1_upr_pred <- pred_m1_pred[, "upr"]
dd$m2_fit <- predict(m2, newdata = dd)

# Построение графика
ggplot(dd, aes(y = Drinks_per_week, x = Party_Hours_per_week, color = Gender)) +
  geom_point() +  
  scale_color_manual(values = c("blue", "red", "green"), labels = c("male", "female", "other")) +
  
  # Линия для модели m1
  geom_line(aes(y = m1_fit), color = "darkblue", linetype = "solid") + # Основная линия
  geom_ribbon(aes(ymin = m1_lwr_conf, ymax = m1_upr_conf), fill = "lightblue", alpha = 0.5) + # Доверительный интервал
  #geom_ribbon(aes(ymin = m1_lwr_pred, ymax = m1_upr_pred), fill = "orange", alpha = 0.3) + # Предиктивный интервал
  
  # Линия для модели m2
  #geom_line(aes(y = m2_fit), color = "green", linetype = "dashed") + # Линия для m2
  
  labs(title = "Модель 1: Party Hours per Week vs. Drinks per Week",
       y = "Drinks per Week",
       x = "Party Hours per Week") +
  theme_minimal()



# ==========================
# Предположим, что m1 - ваша модель регрессии и validation_data - ваши данные

# 1. Получение предсказаний с доверительными и предиктивными интервалами
predictions <- predict(m1, newdata = validation_data, interval = "confidence", level = 0.95)
predictions_pred <- predict(m1, newdata = validation_data, interval = "prediction", level = 0.95)

# 2. Создание новых данных для прогнозирования
new_data <- validation_data
new_data$Party_Hours_per_week_5 <- new_data$Party_Hours_per_week * 1.05
new_data$Party_Hours_per_week_10 <- new_data$Party_Hours_per_week * 1.10
new_data$Party_Hours_per_week_15 <- new_data$Party_Hours_per_week * 1.15

# 3. Получение предсказаний для новых данных
predictions_5 <- predict(m1, newdata = new_data, interval = "confidence", level = 0.95)
predictions_10 <- predict(m1, newdata = new_data, interval = "confidence", level = 0.95)
predictions_15 <- predict(m1, newdata = new_data, interval = "confidence", level = 0.95)

# 4. Построение графиков
library(ggplot2)

# Создание основного графика
plot_data <- data.frame(
  Actual = validation_data$Party_Hours_per_week,
  Predicted = predictions[, "fit"],
  Lower_CI = predictions[, "lwr"],
  Upper_CI = predictions[, "upr"],
  Predicted_Pred = predictions_pred[, "fit"],
  Lower_Pred = predictions_pred[, "lwr"],
  Upper_Pred = predictions_pred[, "upr"]
)

ggplot(plot_data, aes(x = Actual)) +
  geom_point(aes(y = Predicted), color = "blue") +
  geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI), alpha = 0.2, fill = "lightblue") +
  geom_ribbon(aes(ymin = Lower_Pred, ymax = Upper_Pred), alpha = 0.2, fill = "lightgreen") +
  labs(title = "Доверительный и предиктивный прогноз",
       x = "Фактические значения Party_Hours_per_week",
       y = "Прогнозируемые значения") +
  theme_minimal()


#====================

