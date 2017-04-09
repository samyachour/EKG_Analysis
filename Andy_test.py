import wave # this is the wave.py file in the local folder
#import challenge
## np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing
import challenge
import numpy as np
import pywt
import warnings
import pandas as pd
warnings.simplefilter("error") # Show warning traceback


#print ('helloworld')
path='../Physionet_Challenge/training2017/'

records_train = pd.read_csv(path+'REFERENCE.csv', names=['file','answer'])
records_validation = pd.read_csv('validation/' + 'REFERENCE.csv', names=['file','answer'])

new_validation = []
for index, row in records_validation.iterrows():
    record = records_train.ix[records_train['file']==row['file']]
    filename = record['file'].tolist()
    answer = record['answer'].tolist()
    new_validation.append([filename[0],answer[0]])
    



#print(len(records))
#print(records)

records = wave.getRecords('All')

wired_list=[]

feat_list=[]
for record in records:
#    try:
    data = wave.load(record)
    print ('running record: '+ record)
    #sig = challenge.Signal(record, data)
    noise_features = challenge.noise_feature_extract(data)
    feat_list.append(noise_features)
    print ('the number of records in the feature list: ' + str(len(feat_list)))
#    except:
#        wired_list.append(record)
#        print ('stupid one found: ' + record)
#    
#
feat_list = np.array(feat_list)


#feat_list=[]
#for record in records:
##    try:
#    data = wave.load(record)
#    print ('running record: '+ record)
#    sig = challenge.Signal(record, data)
#    features = challenge.feature_extract(sig)
#    feat_list.append(features)
#    print ('the number of records in the feature list: ' + str(len(feat_list)))
##    except:
##        wired_list.append(record)
##        print ('stupid one found: ' + record)
##    
##
#feat_list = np.array(feat_list)

#
#for wired_one in wired_list:
#    try:
#        print(wired_one)
#        data = wave.load(wired_one)
#        sig = featureExtraction.Signal(wired_one, data)
#        features, noise_features = challenge.feature_extract(sig)
#        feat_list.append(features)
#    except Exception as e:
#        print(e)
#        sig.plotRPeaks()



#def noise_feature_extract(records, path = '../Physionet_Challenge/training2017/'):
#    """
#    A function takes in a list of records and returns a matrix of features
#
#    Parameters
#    ----------
#        records: the file name of the file containing the record names (string)
#        wavelet: 'sym4'
#        levels: wavelet 5 level decomposition
#        mode: 'symmetric'
#        omission: get rid of D1 and keep cA
#        path: the path to the file
#        
#    Returns
#    -------
#        1. A numpy array of stats for all wavelet coefficients for all the records
#        2. A numpy array of residuals for all the records
#
#    """
#    full_list = []
#    residual_list = []
#    file = open(path+records, 'r')
#    x=0
#    while (True):
#        newline = file.readline().rstrip('\n')
#        if newline == '':
#            break
#        data = wave.load(newline)
#        coeffs = pywt.wavedecn(data, 'sym4', level=5)
#        feat_list = wave.stats_feat(coeffs)
#    
#        #feat_list = feat_combo(feat_list)
#        residual = wave.calculate_residuals(data)
#        residual_list.append(residual)
#        full_list.append(feat_list)
#        x+=1
#        print('working on file '+ newline)
#        print('length of the data:' + str(len(data)))
#        print('feature created, record No.' + str(x))
#        print('length of feature:'+ str(len(feat_list)))
#    file.close()
#    return np.array(full_list), np.array(residual_list)
#
#noise_feature, residual = noise_feature_extract('RECORDS')

