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
        if self.inverted: # flip the inverted signal
            self.data = -data
            
    def plotRPeaks(self):
        plt.plot(self.data) # TODO: make plot bigger
        plt.plot(*zip(*self.RPeaks), marker='o', color='r', ls='')
        plt.title(self.name)
        plt.show()
        
        
    # TODO: add error handling for crazy cases of data i.e. A04244
    # Wrap the whole thing in a try catch, assign as AF if there's an error
    # Set everything to N in the beginning
            

# Run Wavelet transforms

"""
level = 6
omission = ([5,6], True)

wave.plot(data, "Original Signal", "Index n * 0.003")
rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
wave.plot(rebuilt, omission, "Index n * 0.003")
"""
    

# Imperatively grabbing features

# Testing P wave detection

records = wave.getRecords('N') # N O A ~
data = wave.load(records[7])
sig = Signal(records[7],data)

sig.plotRPeaks()
baseline = wave.getBaseline(sig)

plt.plot(sig.data[0:1000])
plt.axhline(baseline)
plt.show()


# Testing RR Interval calculation

records = wave.getRecords('O') # N O A ~
data = wave.load('A04244')
sig = Signal('A04244',data)

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


# Detecting noise

# noise_feat_mat, residuals = wave.noise_feature_extract('RECORDS')