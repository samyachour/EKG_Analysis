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
    var_PP = np.var(signal.PPinterval)
    
    
    #wavelet decomp coeff
    wtcoeff = pywt.wavedecn(signal.data, 'sym5', level=5, mode='constant')
    wtstats = wave.stats_feat(wtcoeff)
    
    #RR interval
    RRinterval_bin = wave.interval_bin(signal.RRintervals)
    RRinterval_stats = wave.RR_interval_stats(signal.RRintervals)
    RPeak_stats = wave.R_peak_stats(signal.RPeaks)
    
    #variances for every other variances, every third, every fourth
    RR_var_other = wave.var_every_other(signal.RRintervals)
    RR_var_third = wave.var_every_third(signal.RRintervals)
    RR_var_fourth = wave.var_every_fourth(signal.RRintervals)
    RR_var_next = wave.var_next(signal.RRintervals)
    #noise features:
    residuals = wave.calculate_residuals(signal.data)
    
    feature_list = RRinterval_bin + RRinterval_stats + RPeak_stats + wtstats
    feature_list.append(residuals)
    feature_list.append(RR_var_other)
    feature_list.append(RR_var_next)
    feature_list.append(RR_var_fourth)
    feature_list.append(RR_var_third)
    feature_list.append(var_PP)
    return feature_list

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
