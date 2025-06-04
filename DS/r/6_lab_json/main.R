setwd(
  '/home/ndavs/study/math/DS/r/6_lab_json'
)
library("dplyr")  # манипуляции с данными
library("ggplot2")  # графики

library("jsonlite")

file_path <- './ADR.txt'
aa<-fromJSON(file_path)
is(aa)
aa
aa$addresses


k = 1
l=1
columns <- strsplit(aa$addresses[[k]], ' ')
columns[[l]]
l
l <- ifelse(length(columns[[l]]) > 2, l , l+1)

columns[[l]][7]
cityInd <- which(columns[[l]] == "Владивосток")


streetInd <- cityInd + 2
columns[[l]][streetInd] 

which.min(is.integer(columns[[l]][streetInd:]))
which(is.integer(columns[[l]][streetInd:length(columns[[l]])]))

subset_vector <- columns[[l]][streetInd:length(columns[[l]])]


int_indices <- which(sapply(subset_vector, is.integer))

# Adjusting for the original index if needed
adjusted_indices <- int_indices + streetInd - 1

columns <- unlist(columns)

columns[[2]]
columns[[3]] 
columns[[2]][1]
columns[[2]][2]
columns[[2]][3]
columns[[2]][4]
ind <- which(columns[[2]] == "Владивосток")
ind

colu<-columns[[2]][4]
colu2<-columns[[2]][6]
colu1<-columns[[2]][7]

colu3<-columns[[2]][10]

colu3 <- strsplit(colu3, ",")
colu3 <- unlist(colu3)
colu1
colu1<-'ул.'
adr<-paste(colu,',',colu1, colu2,',', colu3,sep = '', collapse=NULL)
adr
df<-data.frame(id="id1",adress=adr)
df

