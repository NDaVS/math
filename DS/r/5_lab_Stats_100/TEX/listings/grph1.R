library(ggplot2)

x_seq <- seq(
  from = min(train_data$Party_Hours_per_week),
  to = max(train_data$Party_Hours_per_week),
  length.out = 100
)

newdata_m1 <- expand.grid(
  Party_Hours_per_week = x_seq,
  Gender = levels(train_data$Gender)
)
newdata_m1$Home_Town <- NA  # Добавляем для совместимости
newdata_m1$Pred <- predict(m1, newdata_m1)
newdata_m1$Model <- "m1"
newdata_m1$Group <- newdata_m1$Gender

newdata_m2 <- expand.grid(
  Party_Hours_per_week = x_seq,
  Gender = levels(train_data$Gender),
  Home_Town = c(0, 1)
)
newdata_m2$Pred <- predict(m2, newdata_m2)
newdata_m2$Model <- "m2"
newdata_m2$Group <- interaction(newdata_m2$Gender, newdata_m2$Home_Town, sep = "_HT")

pred_all <- rbind(newdata_m1, newdata_m2)
train_data$Home_Town <- factor(train_data$Home_Town)

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
    subtitle = "Цвет - пол, форма - родной город",
    x = "Party Hours per Week",
    y = "Drinks per Week",
    color = "Пол (Gender)",
    shape = "Родной город (Home_Town)"
  ) +
  theme_minimal()

