x_max <- max(dd$Party_Hours_per_week)
x_ext <- x_max * c(1.05, 1.10, 1.15)

new_rows <- expand.grid(
  Party_Hours_per_week = x_ext,
  Gender = levels(train_data$Gender)
)

new_rows <- expand.grid(Party_Hours_per_week = x_ext, Gender = levels(train_data$Gender))

new_rows$Drinks_per_week <- predict(m1, newdata = new_rows)
new_rows$Home_Town <- NA 

conf <- predict(m1, newdata = new_rows, interval = "confidence")
pred <- predict(m1, newdata = new_rows, interval = "prediction")

results <- cbind(new_rows, conf, pred_Lwr = pred[ , "lwr"], pred_Upr = pred[ , "upr"])
print(results)
