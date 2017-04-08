#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:35:46 2017

@author: Work
"""

import wave # this is the wave.py file in the local folder
import featureExtraction 
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing
import challenge


print ('helloworld')

records = wave.getRecords('All') # N O A ~

print(len(records))
print(records)