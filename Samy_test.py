# Playground to try things out/store extra code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import detect_peaks
import wave
import scipy
import plot

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

""" Journal club presentation """

def pointDetection():
    records = wave.getRecords('N') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()
    
    records = wave.getRecords('A') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()
    
    records = wave.getRecords('O') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()
    
    
    records = wave.getRecords('~') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()

# Don't use
def s_decomp(cA, wavelet, levels, omissions=([], False)): # stationary wavelet transform, AKA maximal overlap
    """
    1-dimensional stationary wavelet decompisition and reconstruction

    Parameters
    ----------
    ---Same as as decomp, not including mode---

    Returns
    -------
        1D array of reconstructed data.

    """

    if omissions[0] and max(omissions[0]) > levels:
        raise ValueError("Omission level %d is too high.  Maximum allowed is %d." % (max(omissions[0]), levels))
        
    coeffs = pywt.swt(cA, wavelet, level=levels, start_level=0)
    coeffs = omit(coeffs, omissions, stationary=True)
    
    return pywt.iswt(coeffs, wavelet)

def noiseRemoval():

    data = wave.load('A00269')
    wave.plot(data, title="A00269")
    data = wave.discardNoise(data)
    wave.plot(data, title="A00269 - After noise removal")
    
    level = 6
    omission = ([1,2], True) # 5-40 hz
    rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
    wave.plot(rebuilt, title="A00269 - After wavelet decompisition") 
    
    data = wave.load('A00420')
    wave.plot(data, title="A00420")
    data = wave.discardNoise(data)
    wave.plot(data, title="A00420 - After noise removal")
    rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
    wave.plot(rebuilt, title="A00420 - After wavelet decompisition") 
    
    data = wave.load('A00550')
    wave.plot(data, title="A00550")
    data = wave.discardNoise(data)
    wave.plot(data, title="A00550 - After noise removal")
    rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
    wave.plot(rebuilt, title="A00420 - After wavelet decompisition") 

# data = pd.read_csv('../../../../../Downloads/A00001.csv')[[0]]
# print(data)

def getRPeaksOrig(data, minDistance=150):
    """
    R peak detection in 1 dimensional ECG wave

    Parameters
    ----------
    data : array_like
        1-dimensional array with input signal data
    minDistance : positive integer
        minimum distance between R peaks

    Returns
    -------
        tuple consisting of 2 elements:
        if signal is inverted (bool)
        list of tuple coordinates of R peaks in original signal data
        [(x1,y1), (x2, y2),..., (xn, yn)]

    """
    
    # Standardized best wavelet decompisition choice
    level = 6
    omission = ([1,2], True) # 5-40 hz
    rebuilt = decomp(data, 'sym5', level, omissions=omission)
    
    # Get rough draft of R peaks/valleys
    peaks = detect_peaks(rebuilt, mpd=minDistance, mph=0.05)
    if peaks.size == 0:
        positive_R_first = [0]
    else:
        positive_R_first = [rebuilt[i] for i in np.nditer(peaks)]
    pos_mph = np.mean(positive_R_first)
    
    valleys = detect_peaks(rebuilt, mpd=minDistance, mph=0.05,valley=True)
    if valleys.size == 0:
        negative_R_first = [0]
    else:
        negative_R_first = [rebuilt[i] for i in np.nditer(valleys)]
    neg_mph = abs(np.mean(negative_R_first))
    
    # If the wave isn't inverted
    if pos_mph > neg_mph:
        positive_R = detect_peaks(rebuilt, mpd=minDistance, mph=pos_mph - pos_mph/3)
        coordinates = [(int(i), data[i]) for i in np.nditer(positive_R)]
        inverted = False
    
    # If the wave is inverted
    elif neg_mph >= pos_mph:
        negative_R = detect_peaks(rebuilt, mpd=minDistance, mph=neg_mph - neg_mph/3,valley=True)
        # -data[i] because I invert the signal later, and want positive R peak values
        coordinates = [(int(i), -data[i]) for i in np.nditer(negative_R)]
        inverted = True
    
    return (inverted, coordinates)


class Signal(object):
    """
    An ECG/EKG signal

    Attributes:
        name: A string representing the record name.
        data : 1-dimensional array with input signal data
    """

    def __init__(self, name, data):
        """Return a Signal object whose record name is *name*,
        signal data is *data*,
        R peaks array of coordinates [(x1,y1), (x2, y2),..., (xn, yn)]  is *RPeaks*"""
        self.name = name
        self.orignalData = data
        self.sampleFreq = 1/300

        #self.data = wave.discardNoise(data)
        self.data = data

        RPeaks = wave.getRPeaks(self.data, 150)
        self.RPeaks = RPeaks[1]
        self.inverted = RPeaks[0]
        if self.inverted: # flip the inverted signal
            self.data = -self.data

        PTWaves = wave.getPTWaves(self)
        self.PPintervals = PTWaves[0] * self.sampleFreq
        self.Ppeaks = PTWaves[1]
        self.TTintervals = PTWaves[2] * self.sampleFreq
        self.Tpeaks = PTWaves[3]

        self.baseline = wave.getBaseline(self)

        self.Pheights = [i[1] - self.baseline for i in self.Ppeaks]
        self.Rheights = [i[1] - self.baseline for i in self.RPeaks]

        QSPoints = wave.getQS(self)
        self.QPoints = QSPoints[0]
        self.SPoints = QSPoints[1]
        self.Qheights = [i[1] - self.baseline for i in self.QPoints]
        self.Sheights = [i[1] - self.baseline for i in self.SPoints]
        self.QSdiff = np.asarray(self.Qheights) - np.asarray(self.Sheights)
        self.QSdiff = self.QSdiff.tolist()
        self.QSinterval = np.asarray([i[0] for i in self.SPoints]) - np.asarray([i[0] for i in self.QPoints])
        self.QSinterval = self.QSinterval.tolist()

        # TODO: Get pr and qt, careful with offset

        #RR interval
        self.RRintervals = wave.wave_intervals(self.RPeaks)

def feat_PCA(feat_mat, components=12):
    """
    this function does PCA on a feature matrix

    Parameters
    ----------
        feat_mat: the original matrix
        components: the PCA components we want to keep

    Returns
    -------
        1. PCA components

    """
    pca = PCA(n_components = components)
    pca.fit(feat_mat)
    print('The number of components is: ' + str(components))
    print('The pca explained variance ratio is:' + str(pca.explained_variance_ratio_))
    return pca.components_

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

#def all_F1_score(prediction, target=['N', 'A', 'O', '~'], path='../Physionet_Challenge/training2017/'):
#    output[target]:F1_score(prediction, n, path) }
#    total = 0
#    for i in output:
#        total += i
#    avg = total/4
#    output['avg'] = avg
#    return output



def all_F1_score(prediction, target=['N', 'A', 'O', '~'], path='../Physionet_Challenge/training2017/'):
    for n in target:
        F1 = F1_score(prediction, n, path)

#w = pywt.Wavelet('sym5')
#print(pywt.dwt_max_level(data_len=1000, filter_len=w.dec_len))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = scipy.signal.lfilter(b, a, data)
    y = scipy.signal.filtfilt(b, a, data) # or use b=mexican hat and a=1, resample?
    return y


record = 'A00269'
data = wave.load(record)
plot.plot(data)

level = 6
omission = ([5,6], True) # 5-40 hz

widths = 1
cwtmatr, freqs = pywt.cwt(data, widths, 'mexh')

#plot.plot(cwtmatr)

# Sample rate and desired cutoff frequencies (in Hz).
fs = 300.0
lowcut = 5
highcut = 15

# from physionet sample
b1 = np.asarray([-7.757327341237223e-05,  -2.357742589814283e-04, -6.689305101192819e-04, -0.001770119249103,
                 -0.004364327211358, -0.010013251577232, -0.021344241245400, -0.042182820580118,
                 -0.077080889653194, -0.129740392318591, -0.200064921294891, -0.280328573340852,
                 -0.352139052257134, -0.386867664739069, -0.351974030208595, -0.223363323458050,
                 0, 0.286427448595213, 0.574058766243311, 0.788100265785590, 0.867325070584078,
                 0.788100265785590, 0.574058766243311, 0.286427448595213, 0, -0.223363323458050,
                 -0.351974030208595, -0.386867664739069, -0.352139052257134, -0.280328573340852,
                 -0.200064921294891, -0.129740392318591, -0.077080889653194, -0.042182820580118,
                 -0.021344241245400, -0.010013251577232, -0.004364327211358, -0.001770119249103,
                 -6.689305101192819e-04, -2.357742589814283e-04, -7.757327341237223e-05])

secs = len(b1)/300 # Number of seconds in signal X
samps = secs*250     # Number of samples to downsample
b1 = scipy.signal.resample(b1,samps)
bpfecg = scipy.signal.filtfilt(b1,1,data)

plot.plot(bpfecg)

y = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
plot.plot(y)


#import rpy2.robjects as robjects





