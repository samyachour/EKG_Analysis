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
import matplotlib.pyplot as plt


#coeff_names = generate_name('wavelet_coeff_', 48)
#PP_interval_stats_names = generate_name('PP_interval_stats_', 8)
#PPeaks_stats_names = generate_name('PPeaks_stats_', 8)
#TTinterval_stats_names = generate_name('TTinterval_stats_', 8)
#TPeak_stats_names = generate_name('TPeak_stats', 8)
#RRinterval_stats_names generate_name('')
#RR_var_everyother = wave.diff_var(signal.RRintervals, 2)
#    RR_var_third = wave.diff_var(signal.RRintervals, 3)
#    RR_var_fourth = wave.diff_var(signal.RRintervals, 4)
#    RR_var_next = wave.diff_var(signal.RRintervals, 1)


#TODO: loop add rows of feature vectors, exclude noisy signals


class Signal(object):
    """
    An ECG/EKG signal

    Attributes:
        name: A string representing the record name.
        data : 1-dimensional array with input signal data 
    """

    def __init__(self, name, data):
        """Return a Signal object whose record name is *name*,
        signal data is *data*,
        R peaks array of coordinates [(x1,y1), (x2, y2),..., (xn, yn)]  is *RPeaks*"""
        self.name = name
        self.orignalData = data
        self.sampleFreq = 1/300
        
        self.data = wave.discardNoise(data)
                
        RPeaks = wave.getRPeaks(self.data, 150)
        self.RPeaks = RPeaks[1]
        self.inverted = RPeaks[0]
        if self.inverted: # flip the inverted signal
            self.data = -self.data
        
        PTWaves = wave.getPTWaves(self)
        self.PPintervals = PTWaves[0] * self.sampleFreq
        self.Ppeaks = PTWaves[1]
        self.TTintervals = PTWaves[2] * self.sampleFreq
        self.Tpeaks = PTWaves[3]
        
        self.baseline = wave.getBaseline(self)
        
        self.Pheights = [i[1] - self.baseline for i in self.Ppeaks]
        self.Rheights = [i[1] - self.baseline for i in self.RPeaks]
        
        QSPoints = wave.getQS(self)
        self.QPoints = QSPoints[0]
        self.SPoints = QSPoints[1]
        self.Qheights = [i[1] - self.baseline for i in self.QPoints]
        self.Sheights = [i[1] - self.baseline for i in self.SPoints]
        self.QSdiff = np.asarray(self.Qheights) - np.asarray(self.Sheights)
        self.QSdiff = self.QSdiff.tolist()
        self.QSinterval = np.asarray([i[0] for i in self.SPoints]) - np.asarray([i[0] for i in self.QPoints])
        self.QSinterval = self.QSinterval.tolist()
        
        # TODO: Get pr and qt, careful with offset
        
        #RR interval
        self.RRintervals = wave.wave_intervals(self.RPeaks)
        
            
    def plotRPeaks(self):
        fig = plt.figure(figsize=(9.7, 6)) # I used figures to customize size
        ax = fig.add_subplot(111)
        ax.plot(self.data)
        # ax.axhline(self.baseline)
        ax.plot(*zip(*self.RPeaks), marker='o', color='r', ls='')
        ax.set_title(self.name)
        # fig.savefig('/Users/samy/Downloads/{0}.png'.format(self.name))
        plt.show()        

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

def generate_name_list(name_tuples):
    name_list = []
    for name, n in name_tuples:
        names = generate_name(name, n)
        for ne in names:
            name_list.append(ne)
    return name_list
    

def generate_name(name, n):
    name_list = []
    for i in range(1,n+1):
        create_name = name+str(i)
        name_list.append(create_name)
    return name_list

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
    PPeak_stats = wave.cal_stats([],signal.Pheights)
    
    #TT invervals
    TTinterval_stats = wave.cal_stats([],signal.TTintervals)    
    
    #wavelet decomp coeff
    wtcoeff = pywt.wavedecn(signal.data, 'sym5', level=5, mode='constant')
    wtstats = wave.stats_feat(wtcoeff)
    
    #RR interval
    RRinterval_bin = wave.interval_bin(signal.RRintervals)
    RRinterval_bin_cont = RRinterval_bin[:2]
    RRinterval_bin_dis = RRinterval_bin[3:]
    
    RRinterval_stats = wave.cal_stats([],signal.RRintervals)
    RPeak_stats = wave.cal_stats([],signal.Rheights)
    
    #variances for every other variances, every third, every fourth
    RR_var_everyother = wave.diff_var(signal.RRintervals, 2)
    RR_var_third = wave.diff_var(signal.RRintervals, 3)
    RR_var_fourth = wave.diff_var(signal.RRintervals, 4)
    RR_var_next = wave.diff_var(signal.RRintervals, 1)
    
    # total points
    Total_points = signal.data.size
    
    
    # QS stuff
    QPeak_stats = wave.cal_stats([],signal.Qheights)
    SPeak_stats = wave.cal_stats([],signal.Sheights)
    QSDiff_stats = wave.cal_stats([],signal.QSdiff)
    QSInterval_stats = wave.cal_stats([],signal.QSinterval)
    
    #noise features:
    residuals = wave.calculate_residuals(signal.data)
    
    #
    inverted = signal.inverted
    
    features = wtstats + PPinterval_stats + PPeak_stats + TTinterval_stats + \
                QPeak_stats + SPeak_stats + QSDiff_stats + QSInterval_stats + \
                RRinterval_stats + RPeak_stats + RRinterval_bin_cont
    features.append(RR_var_everyother)
    features.append(RR_var_next)
    features.append(RR_var_fourth)
    features.append(RR_var_third)
    features.append(residuals)
    features.append(Total_points)
    
    features = features + RRinterval_bin_dis
    
    features.append(inverted)
    
#    name_tuples = [('wavelet_coeff_', 48), ('PP_interval_stats_', 8), ('PPeaks_stats_', 8), \
#             ('TTinterval_stats_', 8), ('QPeak_stats_', 8), ('SPeak_Stats_', 8), \
#             ('QSDiff_stats_', 8), ('SQInterval_stats_',8),('RRinterval_stats_', 8), \
#             ('RPeaks_stats_', 8), ('RRinterval_bin_cont_', 3), ('RR_var_', 4), ('Residuals_', 1),  \
#             ('Total_points_', 1),('RRinterval_bin_dis_', 2), ('Inverted_', 1)]

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

#QPeak_stats = wave.cal_stats(signal.Qheights)
#SPeak_stats = wave.cal_stats(signal.Sheights)
#QSDiff_stats = wave.cal_stats(signal.QSdiff)
#QSInterval_stats = wave.cal_stats(signal.QSinterval)

name_tuples = [('wavelet_coeff_', 48), ('PP_interval_stats_', 8), ('PPeaks_stats_', 8), \
             ('TTinterval_stats_', 8), ('QPeak_stats_', 8), ('SPeak_Stats_', 8), \
             ('QSDiff_stats_', 8), ('SQInterval_stats_',8),('RRinterval_stats_', 8), \
             ('RPeaks_stats_', 8), ('RRinterval_bin_cont_', 3), ('RR_var_', 4), ('Residuals_', 1), \
             ('Total_points_', 1), ('RRinterval_bin_dis_', 2), ('Inverted_', 1)]

name_list = generate_name_list(name_tuples)
print(name_list)

records = wave.getRecords('All') # N O A ~
feature_list = []
for record in records:
    data = wave.load(record)
    print ('running record: '+ record)
    sig = Signal(record,data)
    features = feature_extract(sig)
    feature_list.append(features)
    

columns = ['A','B', 'C']
feature_matrix = pd.DataFrame(feature_list, columns=name_list)
