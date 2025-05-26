dd$Home_Town[dd$Home_Town < 2] <- 0
dd$Home_Town[dd$Home_Town >= 2] <- 1
dd$Gender <- factor(dd$Gender)
