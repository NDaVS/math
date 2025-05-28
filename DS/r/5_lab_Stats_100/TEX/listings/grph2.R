range_x <- range(dd$Party_Hours_per_week)
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
  Gender = levels(test_data$Gender)
)

conf_pred <- predict(m1, newdata = newdata, interval = "confidence", level = 0.95)
conf_df <- cbind(newdata, conf_pred)
conf_df$Type <- "Confidence"

pred_pred <- predict(m1, newdata = newdata, interval = "prediction", level = 0.95)
pred_df <- cbind(newdata, pred_pred)
pred_df$Type <- "Prediction"

all_preds <- rbind(conf_df, pred_df)

ggplot() +
  geom_point(data = test_data, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender), alpha = 1) +
  geom_point(data = new_rows, aes(x = Party_Hours_per_week, y = Drinks_per_week, color = Gender, ), alpha = 1) +
  
  geom_line(data = all_preds, aes(x = Party_Hours_per_week, y = fit, color = Gender), size = 1) +
  
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