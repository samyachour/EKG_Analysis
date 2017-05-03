import pywt
import numpy as np
import pandas as pd
import scipy.io as sio
from biosppy.signals import ecg
import scipy
from detect_peaks import detect_peaks as detect_peaks_orig


def getRPeaks(data, sampling_rate=300.):
    """
    R peak detection in 1 dimensional ECG wave
    Parameters
    ----------
    data : array_like
        1-dimensional array with input signal data
    Returns
    -------
    data : array_like
        1_dimensional array with the indices of each peak
    
    """
    
    out = ecg.ecg(data, sampling_rate=sampling_rate, show=False)

    return out[2]


def discardNoise(data, winSize=100):
    """
    Discarding sections of the input signal that are noisy

    Parameters
    ----------
    data : array_like
        1-dimensional array with input signal data
    winSize : int
        size of the windows to keep or discard

    Returns
    -------
    data : array_like
        1-dimensional array with cleaned up signal data

    """
    
    left_limit = 0
    right_limit = winSize
    
    dataSize = data.size
    data = data.tolist()
    
    residuals = []
    
    while True:
        
        if right_limit > dataSize: window = data[left_limit:]
        else: window = data[left_limit:right_limit]
        
        w = pywt.Wavelet('sym4')
        levels = pywt.dwt_max_level(len(window), w)
        
        if levels < 1:
            break
                
        residual = calculate_residuals(np.asarray(window), levels=levels)        
        residuals.append(((left_limit, right_limit),residual))
                
        left_limit += winSize
        right_limit += winSize
    
    cleanData = []
    mean = np.mean([i[1] for i in residuals])
    std = np.std([i[1] for i in residuals])
    
    for i in residuals:
        val = i[1]
        
        if val < mean + std and val > mean - std:
            cleanData += data[i[0][0]:i[0][1]]
    
    #plot.plot([i[1] for i in residuals], title="Residuals", yLab="Residual Stat", xLab=str(winSize) + " sized window")
    
    return np.asarray(cleanData)



def omit(coeffs, omissions, stationary=False):
    """
    coefficient omission

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, {details_level_n}, ... {details_level_1}]
    omissions: tuple(list, bool), optional
        List of DETAIL levels to omit, if bool is true omit cA
    
    Returns
    -------
        nD array of reconstructed data.

    """
    
    for i in omissions[0]:
        coeffs[-i] = {k: np.zeros_like(v) for k, v in coeffs[-i].items()}
    
    if omissions[1]: # If we want to exclude cA
        coeffs[0] = np.zeros_like(coeffs[0])
        
    return coeffs

def decomp(cA, wavelet, levels, mode='constant', omissions=([], False)):
    """
    n-dimensional discrete wavelet decompisition and reconstruction

    Parameters
    ----------
    cA : array_like
        n-dimensional array with input data.
    wavelet : Wavelet object or name string
        Wavelet to use.
    levels : int
        The number of decomposition steps to perform.
    mode : string, optional
        The mode of signal padding, defaults to constant
    omissions: tuple(list, bool), optional
        List of DETAIL levels to omit, if bool is true omit cA

    Returns
    -------
        nD array of reconstructed data.

    """
    
    if omissions[0] and max(omissions[0]) > levels:
        raise ValueError("Omission level %d is too high.  Maximum allowed is %d." % (max(omissions[0]), levels))
        
    coeffs = pywt.wavedecn(cA, wavelet, level=levels, mode=mode)
    coeffs = omit(coeffs, omissions)
    
    return pywt.waverecn(coeffs, wavelet, mode=mode)


def filterSignalMexh(data, sampling_rate=300.0):
    
    """
    bandpass filter using mexican hat hardcoded values from physionet

    Parameters
    ----------
    data : array_like
        1-dimensional array with input data.

    Returns
    -------
        1D array of filtered signal data.

    """
    
    # from physionet sample2017
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
    
    secs = b1.size/sampling_rate # Number of seconds in signal X
    samps = secs*250     # Number of samples to downsample to
    b1 = scipy.signal.resample(b1,int(samps))
    bpfecg = scipy.signal.filtfilt(b1,1,data)
    
    return bpfecg

def filterSignalBios(data, sampling_rate=300.0):
    
    """
    filter signal using biosppy

    Parameters
    ----------
    data : array_like
        1-dimensional array with input data.
    sampling_rate : float, optional
        discrete sampling rate for the signal, physionet training is 300. (hz)

    Returns
    -------
        1D array of filtered signal data.

    """
    
    out = ecg.ecg(data, sampling_rate=sampling_rate, show=False)
    
    return out[1]

def detect_peaks(x, plotX=np.array([]), mph=None, mpd=1, threshold=0, edge='rising', 
                 kpsh=False, valley=False, show=False, ax=None):
    
    """
    Wrapper function for detect_peaks function in detect_peaks.py
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    plotX : 1D array_like optional (default = x)
        original signal you might want to plot detected peaks on, if you used wavelets or the like
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indices of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """
    
    if plotX.size == 0:
        plotX = x # couldn't do in function declaration
    return detect_peaks_orig(x, plotX=plotX, mph=mph, mpd=mpd, threshold=threshold, edge=edge, 
                             kpsh=kpsh, valley=valley, show=show, ax=ax)

def getPWaves(signal):
    """
    P Wave detection

    Parameters
    ----------
    signal : Signal object
        signal object from Signal class in signal.py

    Returns
    -------
    data : array_like
        1_dimensional array with the indices of each p peak
    """
    
    maxesP = []
    
    for i in range(0, len(signal.RPeaks) - 1):
        left_limit = signal.RPeaks[i]
        right_limit = signal.RPeaks[i+1]
        left_limit = right_limit - (right_limit-left_limit)//3
        
        plotData = signal.data[left_limit:right_limit]
        peaks = detect_peaks(plotData, show=True, mpd=160)
                        
        if peaks.size != 0:
            maxesP.append(left_limit + peaks[0]) # need to convert to original signal coordinates
        
    return np.asarray(maxesP)

def getBaseline(signal):
    """
    Baseline estimation

    Parameters
    ----------
    signal : Signal object
        signal object from Signal class in signal.py

    Returns
    -------
        Y value in mV of baseline, float
    """
    
    baselineY = 0
    trueBaselines = 0
    
    for i in range(0, len(signal.RPeaks) - 1):
        left_limit = signal.RPeaks[i]
        right_limit = signal.RPeaks[i+1]

        RRinterval = signal.data[left_limit:right_limit] # the indices for one rrinterval
        innerPeaks = detect_peaks(RRinterval, edge='both', mpd=30) # peaks in between
                
        for i in range(0, len(innerPeaks) - 1):
            # between the first set of peaks
            left_limit = innerPeaks[i]
            right_limit = innerPeaks[i+1]
            
            plotData = RRinterval[left_limit:right_limit]
            
            mean = np.mean(plotData)
            
            bottom_limit = mean - 0.04
            top_limit = mean + 0.04
            
            baseline = True
            
            # if any points in the subinterval are out of the range 'mean +/- 0.04'
            for i in plotData:
                if i < bottom_limit or i > top_limit:
                    baseline = False
            
            if baseline:
                baselineY += mean
                trueBaselines += 1
    
    if trueBaselines > 0:
        return baselineY/trueBaselines
    else:
        return np.mean(signal.data)


""" Helper functions """

def load(filename, path = '../Physionet_Challenge/training2017/'):
    """
    Load signal data in .mat form

    Parameters
    ----------
    filename : String
        The name of the .mat file
    path : String, optional
        The path to the file directory, defaults to physionet training data

    Returns
    -------
    1D array of signal data.
    
    """
    
    mat = sio.loadmat(path + filename + '.mat')
    data = np.divide(mat['val'][0],1000)
    return data

def getRecords(trainingLabel, _not=False, path='../Physionet_Challenge/training2017/REFERENCE.csv'): # N O A ~
    """
    Get record names from a reference.csv

    Parameters
    ----------
    trainingLabel : String
        The label you want to grab, N O A ~ or All
    _not : Bool, optional
        If you want to get everything _except_ the given training label, default False
    path : String, optional
        The path to the Reference.csv, default is the training2017 csv   

    Returns
    -------
        tuple of equally sized lists:
            list of record names
            list of record labels N O A ~
    """
    
    reference = pd.read_csv(path, names = ["file", "answer"]) # N O A ~
    if trainingLabel == 'All':
        return (reference['file'].tolist(), reference['answer'].tolist())
    if _not:
        subset = reference.ix[reference['answer']!=trainingLabel]
        return (subset['file'].tolist(), subset['answer'].tolist())
    else:
        subset = reference.ix[reference['answer']==trainingLabel]
        return (subset['file'].tolist(), subset['answer'].tolist())
    
def partition(index, df):
    """
    Helper function for getPartitionedRecords() function
    Partitions a (subsetted) dataframe into training and testing

    Parameters
    ----------
    index : int 0-9
        The partition section you want to grab for testing, 1 is first 1/10th, 2 is the second 1/10th, etc.
    df : pandas dataframe
        The dataframe of records you want to partition, should have 2 columns 'File' and 'Answer'
        and be all of one class, i.e. all 'Answer's should be 'N'

    Returns
    -------
        tuple of tuples:
            tuple of equally sized lists:
                list of record names for 10% testing data
                list of record labels N O A ~ for 10% testing data
            tuple of equally sized lists:
                list of record names for 90% training data
                list of record labels N O A ~ for 90% training data
    """
    
    size = df.shape[0]
    tenth = int(size * 0.1) # this is a 1/10th of the rows in the dataframe of records
    section = index * tenth
    
    # Grab the section index to 1/10th plus the seciton index
    testing = (df['file'].tolist()[section:section + tenth], 
               df['answer'].tolist()[section:section + tenth])
    
    # Grab the everything but the section->section + 1/10th subset
    training = (df['file'].tolist()[0:section] + df['file'].tolist()[section + tenth:],
                df['answer'].tolist()[0:section] + df['answer'].tolist()[section + tenth:])
    
    return (testing, training)

def getPartitionedRecords(index, path='../Physionet_Challenge/training2017/REFERENCE.csv'): # N O A ~
    """
    Partition all the training data while maintaining the ratios of each class

    Parameters
    ----------
    index : int 0-9
        The partition section you want to grab for testing, 1 is first 1/10th, 2 is the second 1/10th, etc.
    path : String, optional
        The path to the Reference.csv, default is the training2017 csv   

    Returns
    -------
        tuple of tuples:
            tuple of equally sized lists:
                list of record names for 10% testing data
                list of record labels N O A ~ for 10% testing data
            tuple of equally sized lists:
                list of record names for 90% training data
                list of record labels N O A ~ for 90% training data
    """
    
    if index < 0 or index > 9:
        raise ValueError("Index %d is not available, can only partition 10 different ways. Index must be 0-9." % (index))
    
    reference = pd.read_csv(path, names = ["file", "answer"]) # N O A ~
    
    n = reference.ix[reference['answer'] == 'N']
    n = partition(index, n)
    
    o = reference.ix[reference['answer'] == 'O']
    o = partition(index, o)
    
    a = reference.ix[reference['answer'] == 'A']
    a = partition(index, a)
    
    p = reference.ix[reference['answer'] == '~']
    p = partition(index, p)
    
    tempTestRec = []
    tempTestLab = []
    
    tempTrainRec = []
    tempTrainLab = []
    
    for i in [n,o,a,p]:
        
        tempTestRec += i[0][0]
        tempTestLab += i[0][1]
        
        tempTrainRec += i[1][0]
        tempTrainLab += i[1][1]
    
    return ((tempTestRec, tempTestLab),(tempTrainRec, tempTrainLab))
    
    
def interval(data):
    """
    Calculate the intervals from a list

    Parameters
    ----------
    data : array_like
        1-dimensional array with input data.

    Returns
    -------
    intervals : array_like
        an array of interval lengths
    """
    return np.array([data[i+1] - data[i] for i in range(0, len(data)-1)])

def calculate_residuals(original, levels=5):
    # calculate residuals for a single EKG
    """
    Calculate the intervals from a list

    Parameters
    ----------
    original : array_like
        the original signal
    levels : int, optional
        the number of wavelet levels you'd like to decompose to

    Returns
    -------
    residual : float
        the residual value
    """
    rebuilt = decomp(original, wavelet='sym4', levels=levels, mode='symmetric', omissions=([1],False))
    residual = sum(abs(original-rebuilt[:len(original)]))/len(original)
    return residual

def diff_var(intervals, skip=2):
    """
    This function calculate the variances for the differences between 
    each value and the value that is the specified number (skip) 
    of values next to it. eg. skip = 2 means the differences of one value 
    and the value with 2 positions next to it.

    Parameters
    ----------
    intervals : 
        the interval that we want to calculate
    skip : int, optional
        the number of position that we want the differences from

    Returns
    -------
        the variances of the differences in the intervals
    """
    
    diff = []
    for i in range(0, len(intervals)-skip, skip):
        per_diff= intervals[i]-intervals[i+skip]
        diff.append(per_diff)
    diff = np.array(diff)
    return np.var(diff)
    

def interval_bin(intervals, mid_bin_range):
    """
    This function calculate the percentage of intervals that fall
    in certain bins

    Parameters
    ----------
    intervals : array_like
        array of interval lengths
    mid_bin_range: tuple, optional
        edge values for middle bin, defaults to normal record edges

    Returns
    -------
    feat_list : tuple
        tuple of bin values as decimal percentages (i.e. 0.2, 0.6, 0.2)
        (
        percentage intervals below mid_bin_range[0], 
        percentage intervals between mid_bin_range[0] and mid_bin_range[1],
        percentage intervals above mid_bin_range[1]
        )
    """
    
    if len(intervals)==0:
        print('RR interval == 0')
        return [0,0,0]
    
    n_below = 0.0
    n_in = 0.0
    n_higher = 0.0
    
    for interval in intervals:
        
        if interval < mid_bin_range[0]:
            n_below += 1
        elif interval <= mid_bin_range[1]:
            n_in += 1
        else:
            n_higher +=1
    
    feat_list = (n_below/len(intervals), n_in/len(intervals), n_higher/len(intervals))
    
    return feat_list



def cal_stats(data):
    """
     Generate statistics for the data given

    Parameters
    ----------
        data : array_like
            1-dimensional array with input data.
        
    Returns
    -------
        Array of summary statistics

    """
    
    power = np.square(data)
    return np.asarray([
                        np.amin(data),
                        np.amax(data),
                        np.mean(data),
                        np.std(data),
                        np.var(data),
                        np.average(power),
                        np.mean(np.absolute(data))
                      ])
    


def stats_feat(coeffs):
    """
     Generate stats for wavelet coeffcients

    Parameters
    ----------
        coeffs: list
            the wavelet coeffcients with the format [cA, {d:cDn},...,{d:cD1}]
            usually returned from pywt.wavedecn
        
    Returns
    -------
        Array of summary statistics for all coefficients

    """
    #calculate the stats from the coefficients
    features = np.array([])
    features = np.append(features, cal_stats(coeffs[0]))
    
    for i in range(1,len(coeffs)):
        features = np.append(features, cal_stats(coeffs[i]['d']))
        
    return features




