import pandas as pd
import numpy as np
import scipy
import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan)


# Reading in matlab data

mat = scipy.io.loadmat('../Physionet_Challenge/training2017/A00001.mat')
data = np.divide(mat['val'][0],1000)[2:502]

reference = pd.read_csv('../Physionet_Challenge/training2017/REFERENCE.csv', names = ["file", "answer", "SNR"]) # N O A ~
    

# Run Wavelet transforms

wave.plotWave(data, "Original Signal", "Index n * 0.003")

rebuilt = wave.waveletDecomp(data, 'sym5', 5, omissions=([1,3,5], True))
wave.plotWave(rebuilt, "rebuilt", "Index n * 0.003")


# Imperatively grabbing features

# Detecting R Peaks
xMax = np.argmax(rebuilt) # location of max peak
threshold = data[xMax] * 0.35
peaks = np.zeros_like(data)



# Detecting noise

def getRecords(type):
    
    subset = reference.ix[reference['answer']==type]
    return subset

noisy = getRecords("~")
for i, row in noisy.iterrows():
    data = scipy.io.loadmat('../Physionet_Challenge/training2017/{0}.mat'.format(row['file']))
    data = np.divide(data['val'][0],1000)
    # wave.plotWave(data[2:502], "noisy", "index")
    SNR_val = scipy.stats.signaltonoise(data)
    noisy.set_value(i,'SNR',SNR_val)

print(noisy)

print(scipy.stats.signaltonoise(data))
print(scipy.stats.signaltonoise(rebuilt))