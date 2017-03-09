# install.packages("signal")
# library(signal)

ECG <- read.csv("CHOC/MIT-BIH_Arrhythmia/100.csv")

plotData <- ts(ECG[0:2000, c(1,2)])
colnames(plotData) <- c("time", "volts")
head(plotData)

plot.ts(plotData)

# Windowing algorithm i'm going to implement
# http://file.scirp.org/pdf/ABB_2014101714074768.pdf

threshold <- 0.4 * colMax(dat)

rVals <- plotData[plotData[, "volts"] > threshold,]
plot.ts(rVals)

apply(plotData,2,max)