import pandas as pd
import numpy as np
import scipy.stats
import wave
# np.set_printoptions(threshold=np.nan)


# Read in data

EKG = pd.read_csv("../MIT_Data/MIT-BIH_Arrhythmia/100.csv", header=None)
# EKG = pd.read_csv("../MIT_Data/noise/A00585.csv", header=None)

plotData = EKG[2:502]

# x = np.asarray(plotData[0]) # times, every 0.003s
y = np.asarray(pd.to_numeric(plotData[1])) # voltages in mV
    

# Run Wavelet transforms

wave.plotWave(y, "Original Signal", "Index n * 0.003")

rebuilt = wave.waveletDecomp(y, 'sym5', 5, omissions=([1,2,3], True))
wave.plotWave(rebuilt, "rebuilt", "Index n * 0.003")


# Imperatively grabbing features

# Detecting R Peaks

xMax = np.argmax(rebuilt) # location of max peak
threshold = y[xMax] * 0.35




# Detecting noise

print(scipy.stats.signaltonoise(y))
print(scipy.stats.signaltonoise(rebuilt))