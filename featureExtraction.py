import wave # this is the wave.py file in the local folder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing
import warnings
import challenge
warnings.simplefilter("error") # Show warning traceback

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
        
        Pwaves = wave.getPWaves(self)
        self.PPintervals = Pwaves[0] * self.sampleFreq
        self.Ppeaks = Pwaves[1]
        
        self.baseline = wave.getBaseline(self)
        
        self.QSPoints = wave.getQS(self)
        
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
        
    # TODO: add error handling for crazy cases of data i.e. A04244, A00057
    # Wrap the whole thing in a try catch, assign as AF if there's an error
    # Set everything to N in the beginning
    
    # TODO: Write bash script including pip install for pywavelets
        
data = wave.load('A00003')
sig = Signal('A00003', data)
wave.plot(data, title="Original")
fig = plt.figure(figsize=(60, 6)) # I used figures to customize size
ax = fig.add_subplot(211)
ax.plot(sig.data)
ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
ax.plot(*zip(*sig.QSPoints), marker='o', color='r', ls='')
ax.axhline(sig.baseline)
ax.set_title(sig.name)
fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
plt.show()
