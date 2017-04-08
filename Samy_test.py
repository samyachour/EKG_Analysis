# Playground to try things out/store extra code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

'''
EKG = pd.read_csv("../MIT-BIH_Arrhythmia/100.csv", header=None)

plotData = EKG[2:500]

np.set_printoptions(threshold=np.nan)
print(plotData)

x = np.asarray(plotData[0])
y = np.asarray(pd.to_numeric(plotData[1]))

plt.plot(y)
plt.ylabel("mV")
plt.xlabel("Index n * 0.003")
plt.show()
'''


'''
cA4 = coeffs[0]
cD4 = coeffs[1]['d']
cD3 = coeffs[2]['d']
cD2 = coeffs[3]['d']
cD1 = coeffs[4]['d']
#cA4, cD4, cD3, cD2, cD1 = coeffs
'''

'''
plotWave(cD1, "cD1", "Index 2n * 0.003")
plotWave(cD2, "cD2", "Index 3n * 0.003")
plotWave(cD3, "cD3", "Index 4n * 0.003")
plotWave(cD4, "cD4", "Index 5n * 0.003")
plotWave(cA4, "cA4", "Index 5n * 0.003")
'''

'''
Uselessly prints specific detail levels and their corresponding unit conversion back to original
for i in range(1 + 2,levels + 1):
    index = i
    smallK = (levels - i) + 2
    bigK = (levels - i) + 1
    plotWave(coeffs[index]['d'], "cD" + str(bigK), "Index " + str(smallK) + "n * 0.003")
'''

# Don't need anymore since pywavelets is fixed and I can set levels to zeros
# thereby allowing reconstruction to give the full original coordinate system
# ... it was a fun problem to solve though
# Convert x value of level N to original coordinate

def generateLengths(pointsNum, levels):
    lengths = [pointsNum]
    
    for i in range (0,levels):
        if pointsNum % 2 == 0:
            pointsNum = (pointsNum + 6)/2
            lengths.append(pointsNum)
        else:
            pointsNum = (pointsNum + 7)/2
            lengths.append(pointsNum)
    
    return lengths

def getNLevelX(xVal, domain, pointsNum, levels):
    
    lengths = generateLengths(pointsNum, levels)
    index = lengths.index(domain)
    newVal = xVal
    
    while index != 0:
        if lengths[index - 1] % 2 == 0:
            newVal = (newVal * 2) - 6
        else:
            newVal = (newVal * 2) - 7
        index -= 1
    
    return newVal

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
       
# Add arrays by element quickly
def addArrays(arrayList):
    return [sum(x) for x in zip(*arrayList)]

'''
testing signal to noise ratio
noisy = getRecords("~")
for i, row in noisy.iterrows():
    data = scipy.io.loadmat('../Physionet_Challenge/training2017/{0}.mat'.format(row['file']))
    data = np.divide(data['val'][0],1000)
    SNR_val = scipy.stats.signaltonoise(data)
    noisy.set_value(i,'SNR',SNR_val)
'''

'''
plot the records
for index, row in getRecords('A').iterrows():
    mat = scipy.io.loadmat('../Physionet_Challenge/training2017/{0}.mat'.format(row['file']))
    data = np.divide(mat['val'][0],1000)
    data = data[:1000]
    wave.plot(data, "Atrial", "Index")
    if index > 200:
        break
'''

def getRecords(type):
    
    subset = reference.ix[reference['answer']==type]
    return subset

# Testing RR Interval calculation

#records = wave.getRecords('O') # N O A ~
#data = wave.load('A04244')
#sig = Signal('A04244',data)

#sig.plotRPeaks()
#wave.getPWaves(sig)

#RR_interval = wave.RR_interval(sig.RPeaks)
#
#RR_interval_diff = wave.interval(RR_interval)
#
#bin_RR_intervals = wave.RR_interval_bin(RR_interval)
#
#x=0
#for i in range(0, len(records)):
#    print('working record:' + records[i])
#    x+=1
#    data = wave.load(records[i])
#    sig = Signal(records[i],data)
#    
#    RR_interval = wave.RR_interval(sig.RPeaks)
#    
#    RR_interval_diff = wave.interval(RR_interval)
#    
#    bin_RR_intervals = wave.RR_interval_bin(RR_interval)
#    print (bin_RR_intervals)


"""
# Testing QS points
data = wave.load('A00857')
wave.plot(data)
signal = Signal('A00857', data)
signal.plotRPeaks()

fig = plt.figure(figsize=(60, 6))
ax = fig.add_subplot(111)
ax.plot(signal.data)
ax.set_xlim(0, signal.data.size)
ax.plot(*zip(*signal.QSPoints), marker='o', color='r', ls='')
fig.savefig('/Users/samy/Downloads/graph.png')
plt.close()

"""

# data = pd.read_csv('../../../../../Downloads/A00001.csv')[[0]]
# print(data)

"""
More testing
weird_records = ['A00111','A00269','A00420','A00550','A00692','A01053','A01329','A01509','A01650','A01734','A01780','A01980','A02021','A02282','A02397','A02478','A02569','A02777','A02781','A03196','A03581','A03650','A04342','A04378','A04465','A04824','A04979','A05261','A06371','A06471','A06495','A06632','A06697','A06895','A06931','A07016','A07088','A07098','A07235','A07933','A08092','A08327']
'''
for i in weird_records:
    data = wave.load(i)
    sig = Signal(i, data)
    sig.plotRPeaks()
'''
data = wave.load(weird_records[0])
sig = Signal(weird_records[0], data)
sig.plotRPeaks()

level = 6
omission = ([5,6], True) # 5-40 hz
rebuilt = wave.decomp(data, 'sym4', level, omissions=omission)
wave.plot(rebuilt)

data = wave.load('A00002')
#data = pd.read_csv('../../../../../Downloads/A00001.csv')[[0]].as_matrix()
#data = np.asarray([i[0] for i in data])
sig = Signal('A00001', data)
fig = plt.figure(figsize=(400, 6)) # I used figures to customize size
ax = fig.add_subplot(211)
ax.plot(sig.data)
ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
ax.plot(*zip(*sig.QSPoints), marker='o', color='r', ls='')
ax.axhline(sig.baseline)
ax.set_title(sig.name)
fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
plt.show()

'''
records = wave.getRecords('~') # N O A ~

for i in records:
    data = wave.load(i)
    #print ('working on Record:' + i)
    sig = Signal(i,data)

'''
"""
