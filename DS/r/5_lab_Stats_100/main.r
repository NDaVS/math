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
colSums(is.na(dd))
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
#====================
predictions_m1 <- predict(m1, newdata=test_data)
predictions_m2 <- predict(m2, newdata=test_data)
mse_m1 <- mean((test_data$Drinks_per_week - predictions_m1)^2)
mse_m2 <- mean((test_data$Drinks_per_week - predictions_m2)^2)

cat("MSE for Model 1 (m1):", mse_m1, "\n")
cat("MSE for Model 2 (m2):", mse_m2)

#===================
library(ggplot2)

# 1. Диапазон значений по оси X
x_seq <- seq(
  from = min(dd$Party_Hours_per_week),
  to = max(dd$Party_Hours_per_week),
  length.out = 100
)

# 2. Предсказания модели m1
newdata_m1 <- expand.grid(
  Party_Hours_per_week = x_seq,
  Gender = levels(dd$Gender)
)
newdata_m1$Home_Town <- NA  # Добавляем для совместимости
newdata_m1$Pred <- predict(m1, newdata_m1)
newdata_m1$Model <- "m1"
newdata_m1$Group <- newdata_m1$Gender

# 3. Предсказания модели m2
newdata_m2 <- expand.grid(
  Party_Hours_per_week = x_seq,
  Gender = levels(dd$Gender),
  Home_Town = c(0, 1)
)
newdata_m2$Pred <- predict(m2, newdata_m2)
newdata_m2$Model <- "m2"
newdata_m2$Group <- interaction(newdata_m2$Gender, newdata_m2$Home_Town, sep = "_HT")

# 4. Объединяем предсказания
pred_all <- rbind(newdata_m1, newdata_m2)
dd$Home_Town <- factor(dd$Home_Town)
# 5. Общий график
ggplot(dd, aes(x = Party_Hours_per_week, y = Drinks_per_week)) +
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
dd$Gender <- factor(dd$Gender)
predictions <- predict(m2, newdata = dd)
residuals <- dd$Drinks_per_week - predictions

# 2. RSS — сумма квадратов ошибок
RSS <- sum(residuals^2)

# 3. TSS — полная сумма квадратов по y
TSS <- sum((dd$Drinks_per_week - mean(dd$Drinks_per_week))^2)

# 4. SSR — регрессионная сумма квадратов
SSR <- TSS - RSS

# 5. Число параметров (без константы)
k <- length(coef(m1)) - 1
n <- nrow(dd)

# 6. F-статистика
F_statistic <- (SSR / k) / (RSS / (n - k - 1))
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
range_x <- range(dd$Party_Hours_per_week)
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
  Gender = levels(dd$Gender)
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
  geom_point(data = dd, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender), alpha = 1) +
  geom_point(data = new_rows, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender, ), alpha = 1) +
  
  # Линия предсказания
  geom_line(data = all_preds, aes(x = Party_Hours_per_week, y = fit, color = Gender), size = 1) +
  
  # Ленты интервалов
  geom_ribbon(
    data = all_preds,
    aes(x = Party_Hours_per_week, ymin = lwr, ymax = upr, fill = Gender),
    alpha = 0.2
  ) +
  
  facet_wrap(~Type) +
  
  labs(
    title = "Доверительный и предиктивный интервалы модели m1",
    subtitle = "Прогноз с расширением области на 15% по Party_Hours_per_week",
    x = "Party Hours per Week",
    y = "Drinks per Week"
  ) +
  theme_minimal()