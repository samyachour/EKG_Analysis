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
    
def getRPeaks(data, minDistance):
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
        if signal is inverted
        list of tuple coordinates of R peaks in original signal data
        [(x1,y1), (x2, y2),..., (xn, yn)]

    """
    
    # Standardized best wavelet decompisition choice
    level = 6
    omission = ([1,2], True) # 5-40 hz
    rebuilt = decomp(data, 'sym5', level, omissions=omission)
    
    # Get rough draft of R peaks/valleys
    positive_R_first = [rebuilt[i] for i in np.nditer(detect_peaks(rebuilt, mpd=minDistance, mph=0.08))]
    pos_mph = np.mean(positive_R_first)
    
    negative_R_first = [rebuilt[i] for i in np.nditer(detect_peaks(rebuilt, mpd=minDistance, mph=0.08,valley=True))]
    neg_mph = abs(np.mean(negative_R_first))
    
    # If the wave isn't inverted
    if pos_mph > neg_mph:
        positive_R = detect_peaks(rebuilt, mpd=minDistance, mph=pos_mph - pos_mph/3)
        coordinates = [(int(i), data[i]) for i in np.nditer(positive_R)]
        inverted = False
    
    # If the wave is inverted
    elif neg_mph > pos_mph or neg_mph > 0.25:
        negative_R = detect_peaks(rebuilt, mpd=minDistance, mph=neg_mph - neg_mph/3,valley=True)
        # -data[i] because I invert the signal later, and want positive R peak values
        coordinates = [(int(i), -data[i]) for i in np.nditer(negative_R)]
        inverted = True
    
    return (inverted, coordinates)

# TODO: Detecting P and T waves, start using wavelets

def getPWaves(signal):
    """
    P Wave detection

    Parameters
    ----------
    signal : Signal object
        signal object from Signal class in signal.py

    Returns
    -------
        list of tuple coordinates of P peaks in original signal data
        [(x1,y1), (x2, y2),..., (xn, yn)]
        
        2D list of lists of 3 tuple coordinates
        first coordinate is start of P wave
        second coordinate is peak of P wave
        third coordinate is end of P wave
        [[(start1x, start1y), (peak1x, peak1y), (end1x, end1y)], 
        [(start2x, start2y), (peak2x, peak2y), (end2x, end2y)]
        [(startNx, startNy), (peakNx, peakNy), (endNx, endNy)]]
    """
    
    level = 6
    omission = ([1,2], True) # <25 hz
    rebuilt = decomp(signal.data, 'sym5', level, omissions=omission)
    
    maxes = []
    
    for i in range(0, len(signal.RPeaks) - 1):
    # for i in range(len(signal.RPeaks) - 9, len(signal.RPeaks) - 4):
        plotData = rebuilt
        right_limit = signal.RPeaks[i+1][0]
        left_limit = right_limit - 70

        plotData = plotData[left_limit:right_limit]
        peaks = detect_peaks(plotData, plotX=signal.data[left_limit:right_limit])
        
        if peaks.size != 0:
            maxes.append(np.amax(peaks))
        else: # if there is no p wave, flat signal in the interval
            maxes.append(0)
    
    return [(i, signal.data[i]) for i in maxes] # P peak coordinates

    
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
    subset = reference.ix[reference['answer']==trainingLabel]
    return subset['file'].tolist()

def plot(y, title="Signal", xLab="Index", folder = ""):
    plt.plot(y)
    plt.ylabel("mV")
    plt.xlabel(xLab)
    plt.title(title)
    if folder != "":
        plt.savefig(folder + title + ".png")
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

def calculate_residuals(original, wavelets, levels, mode='symmetric', omissions=([],True)):
    # calculate residuals for a single EKG
    rebuilt = decomp(original, wavelets, levels, mode, omissions)
    residual = sum(abs(original-rebuilt[:len(original)]))/len(original)
    return residual

def cal_stats(feat_list, data_array):
    #create a list of stats and add the stats to a list
    
    feat_list.append(np.amin(data_array))
    feat_list.append(np.amax(data_array))
    #feat_list.append(np.median(data_array))
    #feat_list.append(np.average(data_array))
    feat_list.append(np.mean(data_array))
    feat_list.append(np.std(data_array))
    feat_list.append(np.var(data_array))
    power = np.square(data_array)
    feat_list.append(np.average(power))
    feat_list.append(np.mean(power))
    #feat_list.append(np.average(abs(data_array)))
    feat_list.append(np.mean(abs(data_array)))
    return feat_list
    


def stats_feat(coeffs):
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

def noise_feature_extract(records, wavelets='sym4', levels=5, mode='symmetric', omissions=([1],False), path = '../Physionet_Challenge/training2017/'):
    #calculate residuals for all the EKGs
    full_list = []
    residual_list = []
    file = open(path+records, 'r')
    x=0
    while (True):
        newline = file.readline().rstrip('\n')
        if newline == '':
            break
        data = load(newline)
        coeffs = pywt.wavedecn(data, 'sym4', level=5)
        feat_list = stats_feat(coeffs)
    
        #feat_list = feat_combo(feat_list)
        residual = calculate_residuals(data, wavelets, levels, mode, omissions)
        residual_list.append(residual)
        full_list.append(feat_list)
        x+=1
        print('working on file '+ newline)
        print('length of the data:' + str(len(data)))
        print('feature created, record No.' + str(x))
        print('length of feature:'+ str(len(feat_list)))
    file.close()
    return np.array(full_list), np.array(residual_list)

"""RR feature extraction"""

def R_peak_stats(peaks):
    """
    Calculate the statistics for the R peaks

    Parameters
    ----------
        peaks: R peaks with tuples (index, R peaks value)

    Returns
    -------
        A list of 8 different statistics

    """
    values = [i[1] for i in peaks]
    feat_list=[]
    values = np.array(values)
    stats = np.array(cal_stats(feat_list, values))
    return stats

def RR_interval(peaks, sampling_frequency=300):
    """
    Get a list of the RR intervals

    Parameters
    ----------
        peaks: R peaks with tuples (index, R peaks value)

    Returns
    -------
        A list of RR intervals

    """
    unit_distance = 1./300
    RR_list = []
    for i in range(0, len(peaks)-1):
        RR_distance = peaks[i][0] - peaks[i+1][0]
        RR_interval = RR_distance * unit_distance
        RR_list.append(abs(RR_interval))
    return np.array(RR_list)


    
