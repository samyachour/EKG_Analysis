import pywt
import wave
import numpy as np
import pandas as pd
import math
import plot

# TODO: Debug record errors
# TODO: code cleanup/refactoring, add unit tests
# TODO: implement bandpass filter, 5-15hz see old paper pan tompkins like https://github.com/fernandoandreotti/qrsdetector/blob/master/pantompkins_qrs.m

# A03509 RRvar1, RRvar2, RRvar3 NaNs
# A03863 A03812 too
# A00111, A00269, A00420, A00550, A00692, A01053, A01329 noisy sections
# A00123, A00119 single inversion

class Signal(object):
    """
    An ECG/EKG signal

    Attributes:
        name: A string representing the record name.
        data : 1-dimensional array with input signal data
    """

    def __init__(self, name, data):
        """
        Return a Signal object whose record name is *name*,
        signal data is *data*,
        R peaks array of coordinates [(x1,y1), (x2, y2),..., (xn, yn)]  is *RPeaks*
        """
        self.name = name
        self.orignalData = data
        self.sampling_rate = 300. # 300 hz
        self.sampleFreq = 1/300

        #self.data = wave.discardNoise(data) # optimize this
        self.data = data

        RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)
        self.RPeaks = RPeaks[1]
        self.inverted = RPeaks[0]
        if self.inverted: # flip the inverted signal
            self.data = -self.data

        self.RRintervals = wave.wave_intervals(self.RPeaks)


data = wave.load('A03812')
sig = Signal('A03812', data)