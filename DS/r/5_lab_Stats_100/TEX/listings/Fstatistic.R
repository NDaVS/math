predictions <- predict(m1, newdata = dd)

residuals <- dd$Drinks_per_week - predictions
RSS <- sum(residuals^2)

TSS <- sum((dd$Drinks_per_week - mean(dd$Drinks_per_week))^2)

SSR <- TSS - RSS

k <- length(coef(m1)) - 1
n <- nrow(dd)

F_statistic <- (SSR / k) / (RSS / (n - k - 1))
F_statistic