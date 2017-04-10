#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:47:16 2017

@author: Work
"""

import pywt
import wave
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# TODO: Debug record errors

# A03509 RRvar1, RRvar2, RRvar3 NaNs
# A03863 A03812 too

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


""" Journal club presentation """

def pointDetection():
    records = wave.getRecords('N') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()
    
    records = wave.getRecords('A') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()
    
    records = wave.getRecords('O') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()
    
    
    records = wave.getRecords('~') # N O A ~
    data = wave.load(records[0])
    sig = Signal(records[0], data)
    fig = plt.figure(figsize=(200, 6)) # I used figures to customize size
    ax = fig.add_subplot(211)
    ax.plot(sig.data)
    ax.plot(*zip(*sig.Ppeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.Tpeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.RPeaks), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.QPoints), marker='o', color='r', ls='')
    ax.plot(*zip(*sig.SPoints), marker='o', color='r', ls='')
    ax.axhline(sig.baseline)
    ax.set_title(sig.name)
    fig.savefig('/Users/samy/Downloads/{0}.png'.format(sig.name))
    plt.show()


def noiseRemoval():

    data = wave.load('A00269')
    wave.plot(data, title="A00269")
    data = wave.discardNoise(data)
    wave.plot(data, title="A00269 - After noise removal")
    
    level = 6
    omission = ([1,2], True) # 5-40 hz
    rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
    wave.plot(rebuilt, title="A00269 - After wavelet decompisition") 
    
    data = wave.load('A00420')
    wave.plot(data, title="A00420")
    data = wave.discardNoise(data)
    wave.plot(data, title="A00420 - After noise removal")
    rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
    wave.plot(rebuilt, title="A00420 - After wavelet decompisition") 
    
    data = wave.load('A00550')
    wave.plot(data, title="A00550")
    data = wave.discardNoise(data)
    wave.plot(data, title="A00550 - After noise removal")
    rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
    wave.plot(rebuilt, title="A00420 - After wavelet decompisition") 
    


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

def noise_feature_extract(data):
    wtcoeff = pywt.wavedecn(data, 'sym5', level=5, mode='constant')
    wtstats = wave.stats_feat(wtcoeff)
    #noise features:
    residuals = wave.calculate_residuals(data)
    noise_features = [residuals] + wtstats
    #noise_features.append(residuals)
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
    RRinterval_bin_cont = RRinterval_bin[:3]
    RRinterval_bin_dis = RRinterval_bin[3:]

    RRinterval_stats = wave.cal_stats([],signal.RRintervals)
    RPeak_stats = wave.cal_stats([],signal.Rheights)

    #variances for every other variances, every third, every fourth
#    RR_var_everyother = wave.diff_var(signal.RRintervals, 2)
#    RR_var_third = wave.diff_var(signal.RRintervals, 3)
#    RR_var_fourth = wave.diff_var(signal.RRintervals, 4)
#    RR_var_next = wave.diff_var(signal.RRintervals, 1)

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
#    features.append(RR_var_everyother)
#    features.append(RR_var_next)
#    features.append(RR_var_fourth)
#    features.append(RR_var_third)
    features.append(residuals)
    features.append(Total_points)

    features = features + RRinterval_bin_dis

    features.append(int(inverted))

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

    reference = pd.read_csv(path + 'REFERENCE.csv', names= ['file', 'answer'])
    ref_dict = {rows['file']:rows['answer'] for index, rows in reference.iterrows()}


    predict = pd.read_csv(prediction, names = ['file', 'answer'])
    for index, rows in predict.iterrows():
        if ref_dict[rows['file']]==target:
            T+=1

        if rows['answer']==target:
            t += 1
            if ref_dict[rows['file']]==rows['answer']:
                Tt += 1
    print('The target class is: ' + target)
    if T == 0 or t ==0:
        print (target + 'is ' + str(0))
        return 0
    else :
        F1 = 2.* Tt / (T + t)
        print('The F1 score for this class is: ' + str(F1))
        return F1

#def all_F1_score(prediction, target=['N', 'A', 'O', '~'], path='../Physionet_Challenge/training2017/'):
#    output[target]:F1_score(prediction, n, path) }
#    total = 0
#    for i in output:
#        total += i
#    avg = total/4
#    output['avg'] = avg
#    return output



def all_F1_score(prediction, target=['N', 'A', 'O', '~'], path='../Physionet_Challenge/training2017/'):
    for n in target:
        F1 = F1_score(prediction, n, path)

def multi_model(v):

    #get important vectors:
    mb1_mb2 = np.asarray(pd.read_csv('mb1_mb2.csv', header=None))
    mb1_mb2_t = mb1_mb2.T


    B1 = mb1_mb2_t[0] #1 + num of features
    B2 = mb1_mb2_t[1]
    x = np.append(np.asarray([1]), v) #1 + num of features
    print ('x is:' + str(len(x)))
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

    return arryth_type[ind]

def is_noisy(v):
    # exp(t(Beta_Hat)%*%newdata) / (1+exp(t(Beta_Hat)%*%newdata))
    B1 = [-3.836891, 0.16960, -0.39009, -0.13013] #1 + num of features
    #thresh = 0.0219
    thresh = 0.219
    x = [1] + v
    t1 = np.transpose(B1)
    par1 = math.exp(np.dot(t1,x))
    result = (1.0*par1) / (1 + par1)
    print(result)
    return (result > thresh)


def applyPCA(testData, isNoise):
    """
    this function applies PCA to a dataset

    Parameters
    ----------
        testData : 1xN vector (list or numpy array)
            Your feature vector

    Returns
    -------
        A vector of features 1xN

    Notes
    -----
    Code in R:
        ((test.DATA - center.vec)/scale.vec) %*% rotation.matrix

    """

    if isNoise: # if we're doing noisy data PCA, so the first step in get_answer
        # e.g. 1x4
        #get the vectors and matrixs
        pca_matrix = pd.read_csv('noise_pca_matrix.csv', header=None)
        center_scale = np.asarray(pd.read_csv('center_scale.csv', header=None))
        center_scale_t = center_scale.T

        center = center_scale_t[0] # 1xN
        scale = center_scale_t[1] # 1xN
        rotation = np.asarray(pca_matrix) # NxN

    else: # if we're doing regular features PCA, so after noisy signals are disqualified
        # e.g. 1x4
        #geting the important matrix
        multi_pca_matrix = pd.read_csv('multi_pca_matrix.csv', header=None)
        center_scale_multi = np.asarray(pd.read_csv('center_scale_multi.csv', header=None))
        center_scale_multi_t = center_scale_multi.T

        center = np.asarray(center_scale_multi_t[0]) # 1xN
        scale = np.asarray(center_scale_multi_t[1]) # 1xN
        rotation = np.asarray(multi_pca_matrix) # NxN


    testData = np.asarray(testData)

    if center.size == scale.size == testData.size == np.size(rotation,0):
        result = (testData - center)/scale
        return rotation.dot(result)


def get_answer(record, data):
    answer = ""
    try:
        print ('processing record: ' + record)

        print ('noise feature extraction...')
        noise_feature = noise_feature_extract(data)
        ## do PCA here in R
        noise_feature = applyPCA(noise_feature, True)
        PCA_noise_feature = [noise_feature[0], noise_feature[2], noise_feature[4]]

        print ('noise ECG classifier:')
        if is_noisy(PCA_noise_feature):
            answer = "~"
        else:
            print ('Not noisy, initalize signal object...')
            sig = Signal(record, data)

            print ('generating feature vector...')
            features = feature_extract(sig)
            features_cont = features[0:110]
            features_dist = features[110]
            ## do PCA in R
            features_cont = applyPCA(features_cont, False)
            features = np.append(features_cont[0:24],features_dist)
            print ('multinomial classifier:')
            answer = multi_model(features)
    except Exception as e:
        print (str(e))
        answer = 'A'

    print ('The ECG is: ' + answer)
    return answer




"""
PHYSIONET SUBMISSION CODE
Add tar.gz files, remove matplotlib, make sure setup.sh includes library, remove/add DRYRUN, leave out test.py and ipnyb
"""

import sys
import scipy.io

record = sys.argv[1]
#record = 'A00001'
# Read waveform samples (input is in WFDB-MAT format)
mat = scipy.io.loadmat("validation/" + record + ".mat")
#samples = mat_data['val']
data = np.divide(mat['val'][0],1000) # convert to millivolts

answer = get_answer(record, data)

# Write result to answers.txt
answers_file = open("answers.txt", "a")
answers_file.write("%s,%s\n" % (record, answer))
answers_file.close()
