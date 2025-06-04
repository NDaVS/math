predictions_m1 <- predict(m1, newdata=test_data)
predictions_m2 <- predict(m2, newdata=test_data)
mse_m1 <- mean((test_data$Drinks_per_week - predictions_m1)^2)
mse_m2 <- mean((test_data$Drinks_per_week - predictions_m2)^2)

cat("MSE for Model 1 (m1):", mse_m1, "\n")
cat("MSE for Model 2 (m2):", mse_m2)
