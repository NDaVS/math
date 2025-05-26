library(ggplot2)

range_x <- range(train_data$Party_Hours_per_week)
x_min <- range_x[1]
x_max <- range_x[2]
x_range <- x_max - x_min

x_ext <- seq(
  from = x_min,
  to   = x_max + 0.15 * x_range,
  length.out = 200
)

newdata <- expand.grid(
  Party_Hours_per_week = x_ext,
  Gender = levels(train_data$Gender)
)

conf_pred <- predict(m1, newdata = newdata, interval = "confidence", level = 0.95)
conf_df <- cbind(newdata, conf_pred)
conf_df$Type <- "Confidence"

# 4. Прогноз: предиктивный интервал
pred_pred <- predict(m1, newdata = newdata, interval = "prediction", level = 0.95)
pred_df <- cbind(newdata, pred_pred)
pred_df$Type <- "Prediction"

# 5. Объединяем всё
all_preds <- rbind(conf_df, pred_df)
x_max <- max(train_data$Party_Hours_per_week)
x_ext <- x_max * c(1.05, 1.10, 1.15)
new_rows <- expand.grid(
  Party_Hours_per_week = x_ext,
  Gender = levels(train_data$Gender)
)
new_rows <- expand.grid(Party_Hours_per_week = x_ext, Gender = levels(train_data$Gender))
new_rows$Drinks_per_week <- predict(m1, newdata = new_rows)
new_rows$Home_Town <- NA 
train_data_extend <- rbind(train_data, new_rows)

ggplot() +
  geom_point(data = train_data, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender), alpha = 0.4) +
  geom_point(data = new_rows, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender, shape=""), alpha = 1) +
  
  geom_line(data = all_preds, aes(x = Party_Hours_per_week, y = fit, color = Gender), size = 1) +
  
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
