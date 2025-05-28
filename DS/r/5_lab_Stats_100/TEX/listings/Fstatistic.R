predictions <- predict(m2, newdata = test_data)

residuals <- test_data$Party_Hours_per_week - predictions

SST <- sum((test_data$Party_Hours_per_week - mean(test_data$Party_Hours_per_week))^2)

SSE <- sum(residuals^2)

k <- length(coef(m1)) - 1
n <- nrow(test_data)
SSR <- SST - SSE

F_statistic <- (SSR / k) / (SSE / (n - k - 1))

F_statistic