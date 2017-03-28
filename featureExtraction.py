import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt

def plotWave(y, title, xLab, folder = ""):
    plt.plot(y)
    plt.ylabel("mV")
    plt.xlabel(xLab)
    plt.title(title)
    if folder != "":
        plt.savefig(folder + title + ".png")
    plt.show()

EKG = pd.read_csv("../MIT_Data/MIT-BIH_Arrhythmia/100.csv", header=None)

plotData = EKG[2:502]

x = np.asarray(plotData[0]) # times, every 0.003s
y = np.asarray(pd.to_numeric(plotData[1])) # voltages in mV

# Wavelet transforms, using pywavelets

def omit(coeffs, levels):
    newCoeffs = coeffs
    
    for i in levels[0]:
        newCoeffs[-i] = {k: np.zeros_like(v) for k, v in coeffs[-i].items()}
    
    if levels[1]:
        newCoeffs[0] = np.zeros_like(coeffs[0])
    
    return newCoeffs

def waveletDecomp(cA, wavelet, levels, mode, omissions):

    coeffs = pywt.wavedecn(cA, wavelet, level=levels, mode=mode)
    # np.set_printoptions(threshold=np.nan)
    
    rebuilt = pywt.waverecn(coeffs, wavelet, mode=mode)
    plotWave(rebuilt, "Original Signal", "Index n * 0.003")
    
    # Automatically remove certain detail coefficients in coeffs
            
    coeffs = omit(coeffs, omissions)
    return pywt.waverecn(coeffs, wavelet, mode=mode)
    

rebuilt = waveletDecomp(y, 'sym4', 6, 'constant', ([1,2,6], True))
plotWave(rebuilt, "rebuilt", "Index n * 0.003")

# Imperatively grabbing features

# grab max and see if it matches with original signal
xMax = np.argmax(rebuilt)

