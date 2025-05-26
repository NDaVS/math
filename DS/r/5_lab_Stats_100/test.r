set.seed(1)
d1 <- sample(c(0, 1), 50, replace = TRUE)
set.seed(42)
d2 <- sample(c(0, 1), 50, replace = TRUE)

rows_00 <- which(d1 == 0 & d2 == 0)
rows_01 <- which(d1 == 0 & d2 == 1)
rows_10 <- which(d1 == 1 & d2 == 0)
rows_11 <- which(d1 == 1 & d2 == 1)

list(
  counts_00 = length(rows_00), 
  rows_00 = head(rows_00),
  
  counts_01 = length(rows_01), 
  rows_01 = head(rows_01),
  
  counts_10 = length(rows_10), 
  rows_10 = head(rows_10),
  
  counts_11 = length(rows_11), 
  rows_11 = head(rows_11)
)



rows_00 <- seq_along(d1)[d1 == 0 & d2 == 0]
rows_01 <- seq_along(d1)[d1 == 0 & d2 == 1]
rows_10 <- seq_along(d1)[d1 == 1 & d2 == 0]
rows_11 <- seq_along(d1)[d1 == 1 & d2 == 1]

# Выводим количество строк и первые несколько индексов для каждой комбинации
result <- list(
  counts_00 = length(rows_00), 
  rows_00 = rows_00,
  
  counts_01 = length(rows_01), 
  rows_01 = rows_01,
  
  counts_10 = length(rows_10), 
  rows_10 = rows_10,
  
  counts_11 = length(rows_11), 
  rows_11 = head(rows_11)
)

print(result)

library(ggplot2)
data <- data.frame(Index = seq_along(d1), True = d1, Predicted = d2)

# Определяем неправильные предсказания
data$WrongPrediction <- ifelse(data$True != data$Predicted, TRUE, FALSE)

# Создаем scatter plot
ggplot(data, aes(x = Index, y = Predicted)) +
  geom_point(aes(color = WrongPrediction), size = 3) +
  geom_point(data = subset(data, WrongPrediction == TRUE), aes(x = Index, y = Predicted), shape = 4, size = 5, color = "red") +
  scale_y_continuous(breaks = c(0, 1)) +  # Устанавливаем только 0 и 1 на оси Y
  labs(title = "Scatter Plot of Predicted Values by Row Index",
       x = "Row Index",
       y = "Predicted Values (0 or 1)") +
  scale_color_manual(values = c("black", "red"), labels = c("Correct", "Incorrect")) +
  theme_minimal()


wrong_indices <- which(d1 != d2)

# Выводим номера строк с ошибками
print(wrong_indices)