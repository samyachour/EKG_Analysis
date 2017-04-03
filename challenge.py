#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:47:16 2017

@author: Work
"""

import csv

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
