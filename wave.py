import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
import pandas as pd
from detect_peaks import detect_peaks as detect_peaks_orig

""" Wave manipulation and feature extraction """

def omit(coeffs, omissions, stationary=False):
    """
    coefficient omission

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, {details_level_n}, ... {details_level_1}]
    omissions: tuple(list, bool), optional
        List of DETAIL levels to omit, if bool is true omit cA
    stationary : bool, optional
        Bool if true you use stationary wavelet omission, coeffs is [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]

    Returns
    -------
        nD array of reconstructed data.

    """
    
    if stationary: # if we want to use stationary wavelets, which you don't, trust me
        for i in omissions[0]:
            if omissions[1]:
                coeffs[-i] = (np.zeros_like(coeffs[-i][0]), np.zeros_like(coeffs[-i][1]))
            else:
                coeffs[-i] = (np.zeros_like(coeffs[-i][0]), coeffs[-i][1])
        return coeffs
    
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

def discardNoise(data):
    """
    Discarding sections of the input signal that are noisy

    Parameters
    ----------
    data : array_like
        1-dimensional array with input signal data

    Returns
    -------
    data : array_like
        1-dimensional array with cleaned up signal data

    """
    
    left_limit = 0
    right_limit = 200
    
    dataSize = data.size
    data = data.tolist()
    cleanData = []
    
    while True:
        
        if right_limit > dataSize: window = data[left_limit:]
        else: window = data[left_limit:right_limit]
        
        if len(window) < 50:
            cleanData += window
            break
                
        w = pywt.Wavelet('sym4')
        residual = calculate_residuals(np.asarray(window), pywt.dwt_max_level(len(window), w))
        
        
        if residual <= 0.001 and np.std(window) < 1:
            cleanData += window
                
        left_limit += 200
        right_limit += 200
    
    return np.asarray(cleanData)
        
def getRPeaks(data, minDistance=150):
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

# TODO: Make differnt RPeak detection that uses windowing to ignore noisy sections

def getPWaves(signal):
    """
    P Wave detection

    Parameters
    ----------
    signal : Signal object
        signal object from Signal class in signal.py

    Returns
    -------
        tuple consisting of 2 elements:
        P peak to P peak intervals [1,2,1,3,...]
        list of tuple coordinates of P peaks in original signal data [(x1,y1), (x2, y2),..., (xn, yn)]
    """
    
    maxes = []
    
    for i in range(0, len(signal.RPeaks) - 1):
        left_limit = signal.RPeaks[i][0]
        right_limit = signal.RPeaks[i+1][0]
        left_limit = right_limit - (right_limit-left_limit)//3
        # left_limit = right_limit - 70 # 0.21s, usual max length of PR interval

        
        plotData = signal.data[left_limit:right_limit]
        peaks = detect_peaks(plotData, plotX=signal.data[left_limit:right_limit])
        peakYs = [plotData[i] for i in peaks] # to get max peak
        
        if peaks.size != 0:
            maxes.append(left_limit + peaks[np.argmax(peakYs)]) # need to convert to original signal coordinates
        else: # if there is no p wave, flat signal in the interval
            maxes.append(0)
            
    PPintervals = interval(maxes)
    
    return (PPintervals, [(i, signal.data[i]) for i in maxes]) # P peak coordinates


def getQS(signal):
    """
    Q S points detection

    Parameters
    ----------
    signal : Signal object
        signal object from Signal class in signal.py

    Returns
    -------
        list of tuple coordinates, Q and S, for every QRS complex
        [(qx1, qy1), (sx2, sy2), (qx3, qy3), (sx4, sy4)]
    """
    
    QSall = []
    maxData = len(signal.data)
    maxRPeak = len(signal.RPeaks)
        
    for i in range(0, maxRPeak - 1):
        
        RPeak = signal.RPeaks[i][0]
        left_limit = RPeak - 20
        if left_limit < 0: left_limit = 0
        right_limit = RPeak + 20
        if right_limit >= maxData: right_limit = maxData - 1

        RPeakIsol = signal.data[left_limit:right_limit]
        maxIdx = RPeakIsol.size
        middle = maxIdx//2
        delta = 16
        
        QPoint = RPeakIsol[middle-delta:middle]
        if middle-delta < 0: QPoint = RPeakIsol[:middle]
        SPoint = RPeakIsol[middle:middle+delta]
        if middle+delta > maxIdx: SPoint = RPeakIsol[middle:]
        
        Q = detect_peaks(QPoint, mpd = 10, valley=True)
        S = detect_peaks(SPoint, mpd = 10, valley=True)
        
        if Q.size == 0:
            Q = np.argmin(QPoint)
        else:
            Q = Q[0] # get first valley from detect_peaks return array
        if S.size == 0:
            S = np.argmin(SPoint)
        else:
            S = S[0]
        
        # Convert to original signal coordinates
        Q = int(Q + (middle-delta) + left_limit)
        if middle-delta < 0: Q = int(Q + left_limit)
        S = int(S + middle + left_limit)
        QSall.append((Q, signal.data[Q]))
        QSall.append((S, signal.data[S]))
    
    return QSall

def getBaseline(signal):
    """
    Baseline estimation

    Parameters
    ----------
    signal : Signal object
        signal object from Signal class in signal.py

    Returns
    -------
        tuple consisting of two elements:
        Y value in mV of baseline
    """
    
    baselineY = 0
    trueBaselines = 0
    
    for i in range(0, len(signal.RPeaks) - 1):
        left_limit = signal.RPeaks[i][0]
        right_limit = signal.RPeaks[i+1][0]

        RRinterval = signal.data[left_limit:right_limit]
        innerPeaks = detect_peaks(RRinterval, edge='both', mpd=30)
                
        for i in range(0, len(innerPeaks) - 1):
            left_limit = innerPeaks[i]
            right_limit = innerPeaks[i+1]
            
            plotData = RRinterval[left_limit:right_limit]
            
            mean = np.mean(plotData)
            
            bottom_limit = mean - 0.04
            top_limit = mean + 0.04
            
            baseline = True
            
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
    #
    ### A helper function to load data
    # input:
    #   filename = the name of the .mat file
    #   path = the path to the file
    # output:
    #   data = data output
    
    mat = sio.loadmat(path + filename + '.mat')
    data = np.divide(mat['val'][0],1000)
    return data

def getRecords(trainingLabel): # N O A ~
    
    reference = pd.read_csv('../Physionet_Challenge/training2017/REFERENCE.csv', names = ["file", "answer"]) # N O A ~
    if trainingLabel == 'All':
        return reference['file'].tolist()
    else:
        subset = reference.ix[reference['answer']==trainingLabel]
        return subset['file'].tolist()

def plot(y, title="Signal", xLab="Index * 0.003s"):
    fig = plt.figure(figsize=(9.7, 6)) # I used figures to customize size
    ax = fig.add_subplot(111)
    ax.plot(y)
    ax.set_title(title)
    # fig.savefig('/Users/samy/Downloads/{0}.png'.format(self.name))
    ax.set_ylabel("mV")
    ax.set_xlabel(xLab)
    plt.show()

def multiplot(data, graph_names):
    #plot multiple lines in one graph
    # input:
    #   data = list of data to plot
    #   graph_names = list of record names to show in the legend
    for l in data:
        plt.plot(l)
    plt.legend(graph_names)
    plt.show()
    
def interval(data):
    """
    Calculate the intervals from a list

    Parameters
    ----------
    data : a list of data

    Returns
    -------
        a list of intervals
    """
    return np.array([data[i+1] - data[i] for i in range(0, len(data)-1)])
    
""" Noise feature extraction """    

def calculate_residuals(original, levels=5):
    # calculate residuals for a single EKG
    """
    Calculate the intervals from a list

    Parameters
    ----------
    original: the original signal

    Returns
    -------
        the residual
    """
    rebuilt = decomp(original, wavelet='sym4', levels=levels, mode='symmetric', omissions=([1],False))
    residual = sum(abs(original-rebuilt[:len(original)]))/len(original)
    return residual

def cal_stats(feat_list, data_array):
    """
     # Generate statistics for the data given and append each stats to the feature list.
     # Input an empty list if your feature list has nothing at the time

    Parameters
    ----------
        feat_list: the feature list that you want to add the stats to
        data_array: the data you want to generate statistics from
        
    Returns
    -------
        The feat_list with all the statistics added

    """
    #create a list of stats and add the stats to a list
    
    feat_list.append(np.amin(data_array))
    feat_list.append(np.amax(data_array))
    feat_list.append(np.mean(data_array))
    feat_list.append(np.std(data_array))
    feat_list.append(np.var(data_array))
    power = np.square(data_array)
    feat_list.append(np.average(power))
    feat_list.append(np.mean(power))
    feat_list.append(np.mean(abs(data_array)))
    return feat_list
    


def stats_feat(coeffs):
    """
     # Generate stats for wavelet coeffcients
     # This special function is for wavelet coefficients input generated from
     # pywt library because the coeffcients comes in with wired format

    Parameters
    ----------
        coeffs: the wavelet coeffcients with the format [cA, {d:cDn},...,{d:cD1}]
        
    Returns
    -------
        The feat_list with all the statistics added

    """
    #calculate the stats from the coefficients
    feat_list = []
    feat_list = cal_stats(feat_list, coeffs[0])
    for i in range(1,len(coeffs)):
        feat_list = cal_stats(feat_list, coeffs[i]['d'])
    return feat_list

def feat_combo(feat_list):
    #Calculate the combination of each elements for ratios and multilications
    new_list = []
    for i in range (0, len(feat_list)):
        new_list.append(feat_list[i])
    
    for i in range(0, len(feat_list)):
        for j in range(0, len(feat_list)):
            if i != j:
                multiply = feat_list[i]*feat_list[j]
                new_list.append(multiply)
                ratio = feat_list[i]/feat_list[j]
                new_list.append(ratio)
    return new_list

def normalize(feat_list):
    return preprocessing.normalize(feat_list)


"""RR feature extraction"""

def peak_stats(peaks):
    """
    Calculate the statistics for the peaks

    Parameters
    ----------
        peaks: peaks with tuples (index, R peaks value)

    Returns
    -------
        A list of 8 different statistics

    """
    values = [i[1] for i in peaks]
    feat_list=[]
    values = np.array(values)
    stats = cal_stats(feat_list, values)
    return stats


def wave_intervals(peaks, sampling_frequency=300):
    """
    Get a list of intervals

    Parameters
    ----------
        peaks: peaks with tuples (index, R peaks value)

    Returns
    -------
        A list of intervals

    """
    unit_distance = 1./300
    interval_list = []
    for i in range(0, len(peaks)-1):
        distance = peaks[i][0] - peaks[i+1][0]
        interval = distance * unit_distance
        interval_list.append(abs(interval))
    return np.array(interval_list)

def interval_bin(intervals, mid_bin_range=[0.6,1]):
    """
    This function calculate the percentage of intervals that fall under 0.6, between 0.6 and 1, and above 1

    Parameters
    ----------
        intervals: the interval that we wanted to bin
        mid_bin_range: specify the bin range

    Returns
    -------
        feat_list: [percentage intervals below mid_bin_range[0], percentage intervals between mid_bin_range[0]
                    and mid_bin_range[1], percentage intervals above mid_bin_range[1], the index of the bin has 
                    that has the highest percentage, 1 if the third bin is above 0.3]
    """
    n_below = 0
    n_in = 0
    n_higher = 0
    for interval in intervals:
        if interval < mid_bin_range[0]:
            n_below += 1
        elif interval <= mid_bin_range[1]:
            n_in += 1
        else:
            n_higher +=1
    if len(intervals)==0:
        print('RR interval == 0')
    feat_list = [n_below/len(intervals), n_in/len(intervals), n_higher/len(intervals)]
    feat_list.append(feat_list.index(max(feat_list)))
    if feat_list[2] > 0.3:
        feat_list.append(1)
    else:
        feat_list.append(0)
    
    return feat_list

def diff_var(intervals, skip=2):
    """
    This function calculate the variances for the differences between each value and the value that
    is the specified number (skip) of values next to it.
    eg. skip = 2 means the differences of one value and the value with 2 positions next to it.

    Parameters
    ----------
        intervals: the interval that we want to calculate
        skip: the number of position that we want the differences from

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
    
    
    
    
    
