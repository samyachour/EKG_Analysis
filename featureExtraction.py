import pandas as pd
import numpy as np
import scipy
import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan)


# Reading in matlab data

mat = scipy.io.loadmat('../Physionet_Challenge/training2017/A00001.mat')
data = np.divide(mat['val'][0],1000)[2:1002]
    

# Run Wavelet transforms

wave.plotWave(data, "Original Signal", "Index n * 0.003")

rebuilt = wave.waveletDecomp(data, 'sym5', 5, omissions=([1,5], True))
wave.plotWave(rebuilt, "rebuilt", "Index n * 0.003")


# Imperatively grabbing features

# Detecting R Peaks
xMax = np.argmax(rebuilt) # location of max peak
threshold = data[xMax] * 0.35
peaks = np.zeros_like(data)



# Detecting noise

def getRecords(type):
    
    reference = pd.read_csv('../Physionet_Challenge/training2017/REFERENCE.csv') # N O A ~
    subset = reference.ix[df[1]==type]
    return subset

print(getRecords("~"))
print(scipy.stats.signaltonoise(data))
print(scipy.stats.signaltonoise(rebuilt))