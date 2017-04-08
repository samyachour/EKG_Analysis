import wave # this is the wave.py file in the local folder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing
# import warnings
# warnings.simplefilter("error") # Show runtime warning traceback
import challenge

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
        
        self.data = wave.discardNoise(data)
                
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
        self.Rheights = [i[1] - self.baseline for i in self.Rpeaks]
        
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
        
            
    def plotRPeaks(self):
        fig = plt.figure(figsize=(9.7, 6)) # I used figures to customize size
        ax = fig.add_subplot(111)
        ax.plot(self.data)
        # ax.axhline(self.baseline)
        ax.plot(*zip(*self.RPeaks), marker='o', color='r', ls='')
        ax.set_title(self.name)
        # fig.savefig('/Users/samy/Downloads/{0}.png'.format(self.name))
        plt.show()        
        
data = wave.load('A00011')
sig = Signal('A00011', data)
wave.plot(data, title="Original")
fig = plt.figure(figsize=(60, 6)) # I used figures to customize size
ax = fig.add_subplot(211)
ax.plot(sig.data)
ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
ax.axhline(sig.baseline)
ax.axhline(0)
ax.set_title(sig.name)
fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
plt.show()

