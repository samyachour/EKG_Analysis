import wave # this is the wave.py file in the local folder

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
        R peaks array is *RPeaks*"""
        self.name = name
        self.data = data
        self.RPeaks = wave.getRPeaks(data, 150)