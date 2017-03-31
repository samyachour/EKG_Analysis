import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

def plot(y, title, xLab="index", folder = ""):
    plt.plot(y)
    plt.ylabel("mV")
    plt.xlabel(xLab)
    plt.title(title)
    if folder != "":
        plt.savefig(folder + title + ".png")
    plt.show()

# Wavelet transforms

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
    Same as as decomp, not including mode
    omissions: tuple(list, bool), optional
        List of levels D to omit, bool is still cA

    Returns
    -------
        1D array of reconstructed data.

    """

    if omissions[0] and max(omissions[0]) > levels:
        raise ValueError("Omission level %d is too high.  Maximum allowed is %d." % (max(omissions[0]), levels))
        
    coeffs = pywt.swt(cA, wavelet, level=levels, start_level=0)
    coeffs = omit(coeffs, omissions, stationary=True)
    
    return pywt.iswt(coeffs, wavelet)

##helper functions
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

def getRecords(type):
    
    reference = pd.read_csv('../Physionet_Challenge/training2017/REFERENCE.csv', names = ["file", "answer"]) # N O A ~
    subset = reference.ix[reference['answer']==type]
    return subset['file'].tolist()

def multiplot(data, graph_names):
    #plot multiple lines in one graph
    # input:
    #   data = list of data to plot
    #   graph_names = list of record names to show in the legend
    for l in data:
        plt.plot(l)
    plt.legend(graph_names)
    plt.show()
    
def calculate_residuals(original, wavelets, levels, mode='symmetric', omissions=([],True)):
    # calculate residuals for a single EKG
    rebuilt = decomp(original, wavelets, levels, mode, omissions)
    residual = sum(abs(original-rebuilt[:len(original)]))/len(original)
    return residual
    

def noise_feature_extract(records, wavelets='sym4', levels=5, mode='symmetric', omissions=([1],False), path = '../Physionet_Challenge/training2017/'):
    #calculate residuals for all the EKGs
    residual_list = []
    file = open(path+records, 'r')
    while (True):
        newline = file.readline().rstrip('\n')
        if newline == '':
            break
        data = load(newline)
        residuals = calculate_residuals(data, wavelets, levels, mode, omissions)
        residual_list.append(residuals)
    file.close()
    return residual_list

