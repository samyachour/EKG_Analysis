import pywt
import numpy as np
import pandas as pd
import scipy.io as sio
from biosppy.signals import ecg


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
    
    out = ecg.hamilton_segmenter(data, sampling_rate=sampling_rate)
    # or just use ecg.ecg and make out -> out[2]

    return out


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
        residual = calculate_residuals(np.asarray(window), levels=pywt.dwt_max_level(len(window), w))
        
        
        if residual <= 0.001 and np.std(window) < 1:
            cleanData += window
                
        left_limit += 200
        right_limit += 200
    
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

def getRecords(trainingLabel, _not=False): # N O A ~
    
    reference = pd.read_csv('../Physionet_Challenge/training2017/REFERENCE.csv', names = ["file", "answer"]) # N O A ~
    if trainingLabel == 'All':
        return reference['file'].tolist()
    if _not:
        subset = reference.ix[reference['answer']!=trainingLabel]
        return subset['file'].tolist()
    else:
        subset = reference.ix[reference['answer']==trainingLabel]
        return subset['file'].tolist()
    
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