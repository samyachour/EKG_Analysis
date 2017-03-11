import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
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

EKG = pd.read_csv("../MIT-BIH_Arrhythmia/100.csv", header=None, names=["time", "volts"])

plotData = EKG[:500]

x = np.asarray(plotData['time'])
y = np.asarray(plotData['volts'])

#x_smooth = np.linspace(x.min(), x.max(), 4000)
#y_smooth = spline(x, y, x_smooth)

plt.plot(x,y)
plt.ylabel("mV")
plt.xlabel("Time")
plt.show()
# ignore time values

# Wavelet transforms, using 

def getGraphs(waveletType):
    waveletType = waveletType
    w = pywt.Wavelet(waveletType)
    cA = y
    plotWave(cA, "Original", "Index n * 0.003", waveletType + "/")
    
    for i in range(1,5):
        cA, cD = pywt.dwt(cA, wavelet=w, mode='constant')
        plotWave(cA, "cA" + str(i), "Index " + str(i + 1) + "n * 0.003", waveletType + "/")
        plotWave(cD, "cD" + str(i), "Index " + str(i + 1) + "n * 0.003", waveletType + "/")

def addArrays(arrayList):
    return [sum(x) for x in zip(*arrayList)]

cA = y
plotWave(cA, "Original", "Index n * 0.003")

coeffs = pywt.wavedecn(cA, 'sym4', level=4, mode='smooth')
cA4 = coeffs[0]
cD4 = coeffs[1]['d']
cD3 = coeffs[2]['d']
cD2 = coeffs[3]['d']
cD1 = coeffs[4]['d']
#cA4, cD4, cD3, cD2, cD1 = coeffs

rebuilt = pywt.waverecn(coeffs, 'sym4', mode='smooth')
plotWave(rebuilt, "rebuilt1", "hopefully correct indices")

#del coeffs[0]
del coeffs[-1]
del coeffs[-1]
rebuilt = pywt.waverecn(coeffs, 'sym4', mode='smooth')
plotWave(rebuilt, "rebuilt2", "hopefully correct indices")

plotWave(cD1, "cD1", "Index 2n * 0.003")
plotWave(cD2, "cD2", "Index 3n * 0.003")
plotWave(cD3, "cD3", "Index 4n * 0.003")
plotWave(cD4, "cD4", "Index 5n * 0.003")
plotWave(cA4, "cA4", "Index 5n * 0.003")

# Imperatively grabbing features

plotData['volts'].values.max()

threshold = 0.4 * plotData['volts'].values.max()


