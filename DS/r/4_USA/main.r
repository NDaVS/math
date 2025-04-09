setwd("/home/ndavs/study/math/DS/r/4_USA")

data <- read.table('Stat100_200_Fall2016_Survey23_combined.txt', header = FALSE, sep=" ")

class(data)
is.data.frame(data)
head(data)
tail(data)

dim(data)

data <- data[ , -1]
data <- data[ , -67]

nrow(data)
ncol(data)

print(data[1, ])

dd <- as.data.frame(data)
head(dd)

colnames(dd) <- paste0("V", 1:ncol(data))

# Исходные имена столбцов
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
new_drinks$Predicted_Party_Hours_Model1 <- predictions_model1[, "fit"]
new_drinks$Lower_CI_Model1 <- predictions_model1[, "lwr"]
new_drinks$Upper_CI_Model1 <- predictions_model1[, "upr"]

# График для модели 1 с новыми точками
ggplot(dd, aes(x = Drinks_per_week, y = Party_Hours_per_week)) +
  geom_point() +  # Точки данных
  geom_smooth(method = "lm", se = TRUE, color = "blue") +  # Линейная модель с доверительными интервалами
  geom_point(data = new_drinks, aes(x = Drinks_per_week, y = Predicted_Party_Hours_Model1), color = "red", size = 3) +
  geom_errorbar(data = new_drinks, aes(x = Drinks_per_week, 
                                       ymin = Lower_CI_Model1, 
                                       ymax = Upper_CI_Model1), 
                color = "red", width = 0.2) +
  labs(title = "Модель 1: Предсказания с доверительными интервалами",
       x = "Drinks per Week",
       y = "Party Hours per Week") +
  theme_minimal()
