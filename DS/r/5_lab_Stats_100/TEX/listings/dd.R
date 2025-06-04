dd <- data[, c("Drinks_per_week", "Party_Hours_per_week", "Gender", "Home_Town")]
head(dd)
dim(dd)
colSums(is.na(dd))
summary(dd)
