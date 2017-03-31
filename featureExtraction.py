import pandas as pd
import numpy as np
import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan)


# Reading in matlab data

mat = wave.load('A00003')
data = mat[:1000]

reference = pd.read_csv('../Physionet_Challenge/training2017/REFERENCE.csv', names = ["file", "answer"]) # N O A ~

# Run Wavelet transforms

wave.plot(data[:500], "Original Signal", "Index n * 0.003")

rebuilt = wave.decomp(data, 'sym4', 5, omissions=([4,5], True))
wave.plot(rebuilt[:500], "rebuilt", "Index n * 0.003")

# Imperatively grabbing features

# Detecting R Peaks
xMax = np.argmax(rebuilt) # location of max peak
threshold = data[xMax] * 0.35
peaks = np.zeros_like(data)
# TODO: Find all the peak intervals using the threshold ad set them into peaks


# Detecting noise

def getRecords(type):
    
    subset = reference.ix[reference['answer']==type]
    return subset


# TODO: Use fourier transforms to detect noisy datasets