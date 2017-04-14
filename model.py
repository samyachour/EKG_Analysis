import pywt
import wave
import pandas as pd
import numpy as np
import math
import plot

# NOW

# TODO: Remove noisy parts of signal
# TODO: Derive bins from normal records

# LATER

# TODO: code cleanup/refactoring, add unit tests
# TODO: Start using rpy2 to work with alex's code to do regression http://rpy.sourceforge.net/rpy2/doc-dev/html/introduction.html
# TODO: Add back in the p wave detection if needed

# TODO: Deal with weird records....
# A03509 RRvar1, RRvar2, RRvar3 NaNs
# A03863 A03812 too
# A00111, A00269, A00420, A00550, A00692, A01053, A01329 noisy sections
# A00123, A00119 single inversion

class Signal(object):
    """
    An ECG/EKG signal

    Attributes:
        name: A string representing the record name.
        data : 1-dimensional array with input signal data
    """

    def __init__(self, name, data):
        """
        Return a Signal object whose record name is *name*,
        signal data is *data*,
        R peaks array of coordinates [(x1,y1), (x2, y2),..., (xn, yn)]  is *RPeaks*
        """
        self.name = name
        self.sampling_rate = 300. # 300 hz
        self.sampleFreq = 1/300
        
        self.data = wave.filterSignal(data)
        self.data = wave.discardNoise(data) # optimize this
        # self.data = data

        self.RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)
        
        self.RRintervals = wave.interval(self.RPeaks)


record = 'A00269'
data = wave.load(record)
plot.plot(data)
sig = Signal(record, data)

coords = [(i, sig.data[i]) for i in np.nditer(sig.RPeaks)]
plot.plotCoords(sig.data, coords)


"""
record = 'A00269'
data = wave.load(record)
plot.plot(data)

sig = Signal(record, data)

coords = [(i, sig.data[i]) for i in np.nditer(sig.RPeaks)]
plot.plotCoords(sig.data, coords)
"""

"""
records = wave.getRecords('All')
training = records[0][:853]
exception = records[0][7675:]


for i in records[0]:
    data = wave.load(i)
    sig = Signal(i, data)
    print("Processing {}".format(i))
    coords = [(i, sig.data[i]) for i in np.nditer(sig.RPeaks)]
    plot.plotCoords(sig.data, coords)
"""

 
    
    

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

    # REDO

    return None

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
