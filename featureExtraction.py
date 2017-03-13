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

EKG = pd.read_csv("../MIT-BIH_Arrhythmia/100.csv", header=None)

plotData = EKG[2:502]

x = np.asarray(plotData[0])
y = np.asarray(pd.to_numeric(plotData[1]))

# Wavelet transforms, using pywavelets

# Automated way to get graphs for different wavelet types and store the images in folders
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
plotWave(cA, "Original", "Index 1n * 0.003")
print(len(cA))

# level is the last index in list - 1
# currLevel of original is totalLevels + 1
# index kn * 0.003 k is (totalLevels - currLevel) + 2
# cDK K is (totalLevels - currLevel) + 1
# max pywavelets level is 6
wavelet = 'db4'
levels = 6
mode = 'zero'
coeffs = pywt.wavedecn(cA, wavelet, level=levels, mode=mode)
# np.set_printoptions(threshold=np.nan)

for i in range(1,levels + 1):
    index = i
    smallK = (levels - i) + 2
    bigK = (levels - i) + 1
    print(len(coeffs[index]['d']))
    plotWave(coeffs[index]['d'], "cD" + str(bigK), "Index " + str(smallK) + "n * 0.003")

rebuilt = pywt.waverecn(coeffs, wavelet, mode=mode)
plotWave(rebuilt, "rebuilt1", "hopefully correct indices")

coeffs.pop(6)
coeffs.pop(5)
coeffs.pop(4)
#coeffs[1] = None
rebuilt = pywt.waverecn(coeffs, wavelet, mode=mode)
plotWave(rebuilt, "rebuilt2", "hopefully correct indices")

# Imperatively grabbing features


