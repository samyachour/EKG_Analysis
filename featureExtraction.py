import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt

def plotWave(y, title, xLab, folder = ""):
    # ignore time values, just use indices, get's iterated by 0.003 starting at 0
    plt.plot(y)
    plt.ylabel("mV")
    plt.xlabel(xLab)
    plt.title(title)
    if folder != "":
        plt.savefig(folder + title + ".png")
    plt.show()

EKG = pd.read_csv("../MIT_Data/MIT-BIH_Arrhythmia/100.csv", header=None)

plotData = EKG[2:502]

x = np.asarray(plotData[0])
y = np.asarray(pd.to_numeric(plotData[1]))

# Wavelet transforms, using pywavelets

cA = y
wavelet = 'sym4'
levels = 6 # max pywavelets level for e.g. sym4 w/ 500 points is 6
mode = 'constant'
coeffs = pywt.wavedecn(cA, wavelet, level=levels, mode=mode)
# np.set_printoptions(threshold=np.nan)

rebuilt = pywt.waverecn(coeffs, wavelet, mode=mode)
plotWave(rebuilt, "Original Signal", "Index n * 0.003")

# Automatically remove certain detail coefficients in coeffs

def omit(coeffs, levels, cA=False):
    newCoeffs = coeffs
    
    for i in levels:
        newCoeffs[-i] = {k: np.zeros_like(v) for k, v in coeffs[-i].items()}
    
    if cA:
        newCoeffs[0] = np.zeros_like(coeffs[0])
    
    return newCoeffs
        
coeffs = omit(coeffs, [1,2,6], True)
rebuilt = pywt.waverecn(coeffs, wavelet, mode=mode)
plotWave(rebuilt, "rebuilt", "Index n * 0.003")


# Imperatively grabbing features

# grab max and see if it matches with original signal
xMax = np.argmax(rebuilt)
print(cA[xMax])
print(cA[np.argmax(cA)])

