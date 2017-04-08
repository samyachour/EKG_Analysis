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
import pandas as pd
import math
from sklearn.decomposition import PCA





columns = ['A','B', 'C']
masterDF = pd.DataFrame(columns=columns)

#TODO: loop add rows of feature vectors, exclude noisy signals

def feat_PCA(feat_mat, components=12):
    """
    this function does PCA on a feature matrix

    Parameters
    ----------
        feat_mat: the original matrix
        components: the PCA components we want to keep

    Returns
    -------
        1. PCA components

    """
    pca = PCA(n_components = components)
    pca.fit(feat_mat)
    print('The number of components is: ' + str(components))
    print('The pca explained variance ratio is:' + str(pca.explained_variance_ratio_)) 
    return pca.components_

def noise_feature_extract(signal):
    wtcoeff = pywt.wavedecn(signal.data, 'sym5', level=5, mode='constant')
    wtstats = wave.stats_feat(wtcoeff)
    #noise features:
    residuals = wave.calculate_residuals(signal.data)
    noise_features = wtstats
    noise_features.append(residuals)
    return noise_features

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
    
    #TT invervals
    TTinterval_stats = wave.cal_stats([],signal.PPintervals)
    TPeak_stats = wave.peak_stats(signal.Tpeak)
    
    
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
    
    features = RRinterval_bin + RRinterval_stats + PPinterval_stats + RPeak_stats + wtstats + \
                PPeak_stats + TTinterval_stats + TPeak_stats
    features.append(residuals)
    features.append(RR_var_everyother)
    features.append(RR_var_next)
    features.append(RR_var_fourth)
    features.append(RR_var_third)

    return features

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

# TODO: add error handling for crazy cases of data i.e. A04244, A00057
# Wrap the whole thing in a try catch, assign as AF if there's an error
# Set everything to N in the beginning
# TODO: check if noisy when giving out feature matrix

# TODO: run multi model on single rows to return value to answers.txt
# TODO: Write bash script including pip install for pywavelets

def multi_model(v):
    B1 = [.3, .6, .8, .9, .5] #1 + num of features
    B2 = [.6, .4, .2, .7, .2]    
    x = [1] + v #1 + num of features
    t1 = np.transpose(B1) 
    t2 = np.transpose(B2)
    par1 = math.exp(np.dot(t1,x))
    par2 = math.exp(np.dot(t2,x))
    cc    = (par1 + par2 + 1)
    probN = 1.0/cc
    probA = (1.0*par1)/cc
    probO = (1.0*par2)/cc
    ind = np.argmax([probN, probA, probO])
    arryth_type = ["N","A","O"]
    
    return arryth_type[ind];

def is_noisy(v):
    # exp(t(Beta_Hat)%*%newdata) / (1+exp(t(Beta_Hat)%*%newdata))
    B1 = [.3, .6, .8, .9, .5] #1 + num of features
    thresh = 0.03
    x = [1] + v
    t1 = np.transpose(B1)
    par1 = math.exp(np.dot(t1,x))
    result = (1.0*par1) / (1 + par1)
    print(result)
    return (result > thresh)
