m1 <- lm("Drinks_per_week ~ Party_Hours_per_week + Gender", data=train_data)
summary(m1)

m2 <-  lm("Drinks_per_week ~ Party_Hours_per_week + Gender + Home_Town", data=train_data)
summary(m2)
