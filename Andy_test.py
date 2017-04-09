import wave # this is the wave.py file in the local folder
#import challenge
## np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing
import challenge
import numpy as np
import pywt
import warnings
warnings.simplefilter("error") # Show warning traceback


print ('helloworld')

records = wave.getRecords('All') # N O A ~

#print(len(records))
#print(records)


#A00397
#A00763
#A01312
#A01429
#A01818
#A02417
#A02706
#A02961
#A03549
#A04244
#A04735
#A06103
#A07524
#A07648
#A07664
#A08034
#A08402

wired_list=[]

feat_list=[]
for record in records:
#    try:
    data = wave.load(record)
    print ('running record: '+ record)
    sig = challenge.Signal(record,data)
    noise_features = challenge.noise_feat_extract(sig)
    feat_list.append(noise_features)
    print ('the number of records in the feature list: ' + str(len(feat_list)))
#    except:
#        wired_list.append(record)
#        print ('stupid one found: ' + record)
#    
#
feat_list = np.array(feat_list)

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

