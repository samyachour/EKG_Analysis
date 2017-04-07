import wave # this is the wave.py file in the local folder
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing

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
        self.data = data
        RPeaks = wave.getRPeaks(data, 150)
        self.RPeaks = RPeaks[1]
        self.inverted = RPeaks[0]
        Pwaves = wave.getPWaves(self)
        self.PPintervals = Pwaves[0]
        self.Ppeaks = Pwaves[1]
        self.baseline = wave.getBaseline(self)
        if self.inverted: # flip the inverted signal
            self.data = -data
            
    def plotRPeaks(self):
        fig = plt.figure(figsize=(9.7, 6))
        ax = fig.add_subplot(111)
        ax.plot(self.data)
        ax.plot(*zip(*self.RPeaks), marker='o', color='r', ls='')
        ax.set_title(self.name)
        plt.show()
        
        
    # TODO: add error handling for crazy cases of data i.e. A04244
    # Wrap the whole thing in a try catch, assign as AF if there's an error
    # Set everything to N in the beginning

# Imperatively grabbing features

records = wave.getRecords('N') # N O A ~
data = wave.load(records[7])
sig = Signal(records[7],data)

sig.plotRPeaks()

# TODO: Detecting Q and S, put in a function
"""
records = wave.getRecords('N') # N O A ~
data = wave.load(records[0])

level = 6
omission = ([1,6], True)
rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
wave.plot(rebuilt, records[0], "Index n * 0.003")

points = wave.getRPeaks(data, 150)[1]
plt.plot(data)
plt.plot(*zip(*points), marker='o', color='r', ls='')
plt.title(records[0])
plt.show()

for i in range(20,40):
    plotData = rebuilt
    left_limit = points[i][0]-50
    right_limit = points[i][0]+50
    if right_limit > plotData.size:
        break
    plotData = plotData[left_limit:right_limit]
    qrs = wave.detect_peaks(plotData, valley=True, show=True)
"""