import wave # this is the wave.py file in the local folder
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing
import warnings
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
        self.data = data
        
        RPeaks = wave.getRPeaks(self.data, 150)
        self.RPeaks = RPeaks[1]
        self.inverted = RPeaks[0]
        if self.inverted: # flip the inverted signal
            self.data = -data
        
        Pwaves = wave.getPWaves(self)
        self.PPintervals = Pwaves[0]
        self.Ppeaks = Pwaves[1]
        
        self.baseline = wave.getBaseline(self)
        
        self.QSPoints = wave.getQS(self)
        
        #RR interval
        self.RRintervals = wave.RR_interval(self.RPeaks)
        #self.RRintervals_bin = wave.RR_intervals
        
        #noise features:
        #self.
        baseline = wave.getBaseline(self)
        self.baseline = baseline[0]
        self.RRIntervalMeanStd = baseline[1] # Standard deviation of all RR interval means
            
    def plotRPeaks(self):
        fig = plt.figure(figsize=(9.7, 6)) # I used figures to customize size
        ax = fig.add_subplot(111)
        ax.plot(self.data)
        # ax.axhline(self.baseline)
        ax.plot(*zip(*self.RPeaks), marker='o', color='r', ls='')
        ax.set_title(self.name)
        # fig.savefig('/Users/samy/Downloads/{0}.png'.format(self.name))
        plt.show()
    
    # TODO: Write generalized functions for 3 bins, max bin, average, and variance
        
        
    # TODO: add error handling for crazy cases of data i.e. A04244, A00057
    # Wrap the whole thing in a try catch, assign as AF if there's an error
    # Set everything to N in the beginning
    
    # TODO: Write bash script including pip install for pywavelets
    
    # TODO: Run PCA and then model coefficients

weird_records = ['A00111','A00269','A00420','A00550','A00692','A01053','A01329','A01509','A01650','A01734','A01780','A01980','A02021','A02282','A02397','A02478','A02569','A02777','A02781','A03196','A03581','A03650','A04342','A04378','A04465','A04824','A04979','A05261','A06371','A06471','A06495','A06632','A06697','A06895','A06931','A07016','A07088','A07098','A07235','A07933','A08092','A08327']

for i in weird_records:
    data = wave.load(i)
    sig = Signal(i, data)
    sig.plotRPeaks()

data = wave.load(weird_records[1])
sig = Signal(weird_records[1], data)
sig.plotRPeaks()

level = 6
omission = ([5,6], True) # 5-40 hz
rebuilt = wave.decomp(data, 'sym4', level, omissions=omission)
wave.plot(rebuilt)

"""
records = wave.getRecords('~') # N O A ~

for i in records:
    data = wave.load(i)
    #print ('working on Record:' + i)
    sig = Signal(i,data)

"""
