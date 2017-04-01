import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing

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
        List of levels (A & D) to omit, bool is still cA

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

def cal_stats(feat_list, data_array):
    #create a list of stats and add the stats to a list
    
    feat_list.append(np.amin(data_array))
    feat_list.append(np.amax(data_array))
    #feat_list.append(np.median(data_array))
    feat_list.append(np.average(data_array))
    feat_list.append(np.mean(data_array))
    feat_list.append(np.std(data_array))
    feat_list.append(np.var(data_array))
    power = np.square(data_array)
    feat_list.append(np.average(power))
    feat_list.append(np.mean(power))
    feat_list.append(np.average(abs(data_array)))
    feat_list.append(np.mean(abs(data_array)))
    return feat_list
    


def stats_feat(coeffs):
    #calculate the stats from teh coefficients
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

