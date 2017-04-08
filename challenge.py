#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:47:16 2017

@author: Work
"""

import csv
import pywt
import wave
import numpy as np


def feature_extract(signal):
    """
    this function extract the features from the attributes of a signal

    Parameters
    ----------
        signal: the signal object

    Returns
    -------
        A vector of features

    """
    
    #variance for PP, average and variance for P amplitude, Bin PP
    
    PPinterval_stats = wave.cal_stats([],signal.PPintervals)
    PPeak_stats = wave.peak_stats(signal.Ppeaks)
    
    
    #wavelet decomp coeff
    wtcoeff = pywt.wavedecn(signal.data, 'sym5', level=5, mode='constant')
    wtstats = wave.stats_feat(wtcoeff)
    
    #RR interval
    RRinterval_bin = wave.interval_bin(signal.RRintervals)
    RRinterval_stats = wave.cal_stats([],signal.RRintervals)
    RPeak_stats = wave.peak_stats(signal.RPeaks)
    
    #variances for every other variances, every third, every fourth
    RR_var_everyother = wave.diff_var(signal.RRintervals, 2)
    RR_var_third = wave.diff_var(signal.RRintervals, 3)
    RR_var_fourth = wave.diff_var(signal.RRintervals, 4)
    RR_var_next = wave.diff_var(signal.RRintervals, 1)
    
    #noise features:
    residuals = wave.calculate_residuals(signal.data)
    noise_features = wtstats
    noise_features.append(residuals)
    
    features = RRinterval_bin + RRinterval_stats + PPinterval_stats + RPeak_stats + wtstats + PPeak_stats
    features.append(residuals)
    features.append(RR_var_everyother)
    features.append(RR_var_next)
    features.append(RR_var_fourth)
    features.append(RR_var_third)

    return features, noise_features 

def F1_score(prediction, target, path='../Physionet_Challenge/training2017/'):
    ## a function to calculate the F1 score
    # input:
    #   prediction = the prediction output from the model
    #   target = a string for the target class: N, A, O, ~
    #   path =  the path to the reference file
    # output:
    #   F1 = the F1 score for the particular class
    
    ref_dict = {}
    Tt = 0
    t = 0
    T = 0
    with open(path+'REFERENCE.csv', mode='r') as f:
        reader = csv.reader(f)
        ref_dict = {rows[0]:rows[1] for rows in reader}
            
    
    with open(prediction, mode='r') as f:
        reader = csv.reader(f)
        for rows in reader:
            if ref_dict[rows[0]]==target:
                T+=1
            
            if rows[1]==target:
                t += 1
                if ref_dict[rows[0]]==rows[1]:
                    Tt += 1
    print('The target class is: ' + target)
    F1 = 2.* Tt / (T + t)
    print('The F1 score for this class is: ' + str(F1))
    
    return F1
