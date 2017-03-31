import pywt
import numpy as np
import matplotlib.pyplot as plt

def plot(y, title, xLab, folder = ""):
    plt.plot(y)
    plt.ylabel("mV")
    plt.xlabel(xLab)
    plt.title(title)
    if folder != "":
        plt.savefig(folder + title + ".png")
    plt.show()

# Wavelet transforms

def omit(coeffs, levels):
    
    for i in levels[0]:
        coeffs[-i] = {k: np.zeros_like(v) for k, v in coeffs[-i].items()}
    
    if levels[1]: # If we want to exclude cA
        coeffs[0] = np.zeros_like(coeffs[0])
    
    return coeffs

def decomp(cA, wavelet, levels, mode='constant', omissions=([], False)):
    
    if omissions[0] and max(omissions[0]) > levels:
        raise ValueError("Omission level %d is too high.  Maximum allowed is %d." % (max(omissions[0]), levels))
        
    coeffs = pywt.wavedecn(cA, wavelet, level=levels, mode=mode)
    coeffs = omit(coeffs, omissions)
    
    return pywt.waverecn(coeffs, wavelet, mode=mode)

def s_decomp(cA, wavelet, levels, omissions=([], False)): # stationary wavelet transform, AKA maximal overlap
    
    if omissions[0] and max(omissions[0]) > levels:
        raise ValueError("Omission level %d is too high.  Maximum allowed is %d." % (max(omissions[0]), levels))
        
    coeffs = pywt.swt(cA, wavelet, level=levels)
    coeffs = omit(coeffs, omissions) # FIX
    
    return pywt.iswt(coeffs, wavelet)