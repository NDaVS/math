setwd("/home/ndavs/study/math/DS/r/4_USA")

data <- read.table('Stat100_200_Fall2016_Survey23_combined.txt', header = FALSE, sep = " ", skip = 1)

data <- data[ , -1]
data <- data[ , -67]

nrow(data)
ncol(data)

print(data[1, ])

dd <- as.data.frame(data)
head(dd)

column_names <- c("Gender", "Gender_ID", "Greek", "Home_Town", "Ethnicity", 
                  "Religion", "Religious", "ACT", "GPA", "Party_Hours_per_week", 
                  "Drinks_per_week", "Sex_Partners", "Relationships", "First_Kiss_Age", 
                  "Favorite_Life_Period", "Hours_call_Parents", "Social_Media", 
                  "Texts", "Good_or_Well", "Expected_Income", "independence_vs_Respect", 
                  "Curiosity_vs_Good_Manners", "Self_reliance_vs_Obedience", 
                  "Considerate_vs_Well_Behaved", "Sum_Traits", "Primary", "President", 
                  "Liberal", "Political_Party", "Grade_vs_Learning", "Parent_Relationship", 
                  "Work_Hours", "Tuition", "Career", "reason", "god", "devil", "heaven", 
                  "hell", "miracles", "angels", "ghosts", "reincarnation", "astrology", 
                  "psychics", "witches", "zombies", "vampires", "UFO", 
                  "afraid_walk_at_night", "debate", "debate_winner", "likelyvote", 
                  "candidate", "environment", "terrorism", "economy", "racism", 
                  "policebrutality", "lawandorder", "genderEquality", "borderSecurity", 
                  "familyValues", "money", "gunRights", "voteParents"
)

# Выводим имена столбцов
print(column_names)
colnames(dd) <- column_names
# Исходные имена столбцов


header <- readLines('Stat100_200_Fall2016_Survey23_combined.txt', n = 1)

cleaned_header <- gsub("\\s*\\([^()]*\\)", "", header)

# Убираем лишние пробелы
cleaned_header <- gsub("\\s+", " ", cleaned_header)
cleaned_header <- trimws(cleaned_header)

# Проверяем результат
vector <- unlist(strsplit(cleaned_header, " "))[-1]

colnames(dd) <- vector

head(dd)

model1 <- lm("Party_Hours_per_week ~ Drinks_per_week", data = dd)
summary(model1)

dd$Gender <- as.factor(dd$Gender)

model2 <- lm("Party_Hours_per_week ~ Drinks_per_week + Gender", data = dd)
summary(model2)

library(ggplot2)
ggplot(dd, aes(x = Drinks_per_week, y = Party_Hours_per_week)) +
  geom_point() +  # Точки данных
  geom_smooth(method = "lm", se = TRUE, color = "blue") +  # Линейная модель
  labs(title = "Модель 1: Party Hours per Week vs. Drinks per Week",
       x = "Drinks per Week",
       y = "Party Hours per Week") +
  theme_minimal()

ggplot(dd, aes(x = Drinks_per_week, y = Party_Hours_per_week, color = Gender)) +
  geom_point() +  # Точки данных
  geom_smooth(method = "lm", se = TRUE) +  # Линейная модель с учетом Gender
  labs(title = "Модель 2: Party Hours per Week vs. Drinks per Week by Gender",
       x = "Drinks per Week",
       y = "Party Hours per Week") +
  theme_minimal()

variables <- dd[, c("Drinks_per_week", "Party_Hours_per_week")]

# Строим матрицу ковариации
cov_matrix <- cov(variables)

# Проверяем результат
print(cov_matrix)

max_drinks <- max(dd$Drinks_per_week, na.rm = TRUE)

# Создать новый объект для предсказания
new_drinks <- data.frame(
  Drinks_per_week = c(max_drinks * 1.05, max_drinks * 1.10, max_drinks * 1.15)
)

# Предсказать значения Party_Hours_per_week для модели 1
predictions_model1 <- predict(model1, newdata = new_drinks, interval = "confidence")

# Добавляем предсказания к новому датафрейму
new_drinks$Party_Hours_per_week <- predictions_model1[, "fit"]
dd$Lower_CI_Model1 <- predictions_model1[, "lwr"]
dd$Upper_CI_Model1 <- predictions_model1[, "upr"]


# Визуализация предсказанных значений для модели 1
ggplot(dd, aes(x = Drinks_per_week, y = Party_Hours_per_week)) +
  geom_point() +  
  geom_smooth(method = "lm", se = TRUE, color = "blue") +  
  geom_point(data = new_drinks, aes(x = Drinks_per_week, y = Predicted_Party_Hours_Model1), color = "red", size = 3) +
  geom_errorbar(data = new_drinks, aes(x = Drinks_per_week, ymin = Lower_CI_Model1, ymax = Upper_CI_Model1), color = "red", width = 0.5) +
  labs(title = "Продолжение модели: Party Hours per Week vs. Drinks per Week",
       x = "Drinks per Week",
       y = "Party Hours per Week",
       color = "Легенда") +
  scale_color_manual(values = c(
    "Исходные данные" = "black",
    "Модель" = "blue",
    "Продолжение модели" = "red",
    "Предсказанные значения" = "red"
  )) +
  theme_minimal()

ggplot(dd, aes(x = Drinks_per_week, y = Party_Hours_per_week)) +
  geom_point() +  
  geom_smooth(method = "lm", se = TRUE, color = "blue") +  
  geom_line(data = new_drinks, aes(x = Drinks_per_week, y = Predicted_Party_Hours_Model1), color = "red", linetype = "dashed") +
  geom_point(data = new_drinks, aes(x = Drinks_per_week, y = Predicted_Party_Hours_Model1), color = "red", size = 3) +
  labs(title = "Продолжение модели: Party Hours per Week vs. Drinks per Week",
       x = "Drinks per Week",
       y = "Party Hours per Week") +
  theme_minimal()


library(ggplot2)
install.packages('psych')
library(psych)
library(dplyr)
dd <- select(dd, "Gender", "Party_Hours_per_week", "Drinks_per_week", "Home_Town")
dd$Gender <- as.factor(dd$Gender)
str(dd)

a <- c("Small_Town", "Medium_City","Big_City_Subrub","Big_City")
x <- sample(a, 100, replace=TRUE)
head(x)

levels <- sort(unique(x))
f <- match(x, levels)
f[1:10]

levels(f) <- as.character(levels)
class(f) <- "factor"
levels(f)[1:10]

levels <- sort(unique(dd$Home_Town))
dd$Home_Town<- match(dd$Home_Town, levels)
levels(dd$Home_Town) <- as.character(levels)
class(dd$Home_Town) <- "factor"
str(dd)

levels <- sort(unique(dd$Home_Town))  # "Berlin", "London", "Paris", "Rome"
dd$Home_Town <- match(dd$Home_Town, levels)  # 1, 3, 2, 3, 4, 2

# Создаем фактор с уровнями "1", "2", "3", "4"
dd$Home_Town <- factor(dd$Home_Town, levels = 1:4, labels = as.character(1:4))

str(dd)
f[1:10]