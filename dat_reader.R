setwd("/Users/samy/Documents/Programming_Stuff/Data_Science/CHOC/EKG_Analysis")

data <- read.csv("samples001.csv", header = TRUE)

#apply(times, 2, )
#time1 <- times[1]
#library(chron)
#times(time1)

library(ggplot2)

# select first 50 rows of time and II lead
times <- data[2:2000, 1]
IILead <- data[2:2000, 3]

x_name <- "times"
y_name <- "IILead"

require(reshape2)
df <- melt(data.frame(times,IILead))
colnames(df) <- c(x_name, y_name)


ggplot(data=df, aes(x=times, y=IILead, group=1), xaxt='n', yaxt='n') + geom_line()
