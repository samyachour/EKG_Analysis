#!/usr/bin/env python3
import model
import scipy
import numpy as np
import pandas as pd


"""
PHYSIONET SUBMISSION CODE
"""

import sys

record = sys.argv[1]
# Read waveform samples (input is in WFDB-MAT format)
mat = scipy.io.loadmat("validation/" + record + ".mat")
#samples = mat_data['val']
data = np.divide(mat['val'][0],1000) # convert to millivolts

answer = model.get_answer(record, data)

# Write result to answers.txt
answers_file = open("answers.txt", "a")
answers_file.write("%s,%s\n" % (record, answer))
answers_file.close()


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