import pywt
import wave
import pandas as pd
import numpy as np
import math
import plot

# NOW

# TODO: Derive bins from normal records

# LATER

# TODO: Submit DRYRUN entry
# TODO: code cleanup/refactoring, add unit tests
# TODO: Start using rpy2 to work with alex's code to do regression http://rpy.sourceforge.net/rpy2/doc-dev/html/introduction.html
# TODO: Add back in the p wave detection if needed

# TODO: Deal with weird records....
# A03509 RRvar1, RRvar2, RRvar3 NaNs
# A03863 A03812 too
# A00111, A00269, A00420, A00550, A00692, A01053, A01329 noisy sections
# A00123, A00119 single inversion

"""
Upon submission:
    -remove import plot from all files
    -make sure setup.sh includes all the right libs
    -make sure dependencies.txt has the right packages
    -run compress.sh, verify it included the right files, Include DRYRUN?
    -make sure entry.zip is formatted correctly
    -(empty setup.sh & add validation folder temprarily) make sure the whole thing runs without errors, delete pycache
"""





class Signal(object):
    """
    An ECG/EKG signal

    Attributes:
        name: A string representing the record name.
        sampling rate/freq: the sampling rate Hz and frequency (float)
        data : 1-dimensional array with input signal data
        RPeaks : array of R Peak indices
        RRintervals : array of RR interval lengths
        RRbins : tuple of bin percents
    """

    def __init__(self, name, data):
        """
        Return a Signal object whose record name is *name*,
        signal data is *data*,
        R peaks array of coordinates [(x1,y1), (x2, y2),..., (xn, yn)]  is *RPeaks*
        """
        self.name = name
        self.sampling_rate = 300. # 300 hz
        self.sampleFreq = 1/300

        self.data = wave.filterSignal(data)
        # self.data = wave.discardNoise(self.data) # optimize this
        # self.data = data

        self.RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)

        self.RRintervals = wave.interval(self.RPeaks)
        self.RRbins = wave.interval_bin(self.RRintervals)


def feature_extract():
    """
    this function extract the features from the attributes of a signal

    Parameters
    ----------
        signal: the signal object

    Returns
    -------
        A vector of features

    """

    records = wave.getRecords('All')

    labels = records[0]
    bin1 = []
    bin2 = []
    bin3 = []
    
    for i in labels:
        data = wave.load(i)
        sig = Signal(i, data)
        bin1.append(sig.RRbins[0])
        bin2.append(sig.RRbins[1])
        bin3.append(sig.RRbins[2])
    
    training = pd.DataFrame({'bin 1': bin1, 'bin2': bin2, 'bin3': bin3, 'record': records[0], 'label': records[1]})
    training.to_csv('training_data')

    return training

feature_extract()


def get_answer(record, data):
    
    answer = ""
    
    return answer