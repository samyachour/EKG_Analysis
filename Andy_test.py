#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:58:16 2017

@author: Work
"""

import pandas as pd
import numpy as np
import scipy
import util
import pywt
import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan)


# Reading in matlab data

data = util.load('A00001')

sliced_data = data[0:1000]

# Run Wavelet transforms

wave.plotWave(sliced_data, "Original Signal", "Index n * 0.003")

single_level_coef = pywt.dwtn(sliced_data, 'sym5')

slvl_coef = pywt.dwt(sliced_data, 'sym5')




#rebuilt = wave.waveletDecomp(sliced_data, 'sym5', 5, omissions=([1,5], True))
#wave.plotWave(rebuilt, "rebuilt", "Index n * 0.003")


# Imperatively grabbing features

# Detecting R Peaks
#xMax = np.argmax(rebuilt) # location of max peak
#threshold = data[xMax] * 0.35
#peaks = np.zeros_like(data)



# Detecting noise
