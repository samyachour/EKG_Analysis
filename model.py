import wave
import numpy as np
import pandas as pd
import pickle
import pywt

# NOW

# TODO: Start using rpy2 to work with alex's code to do regression http://rpy.sourceforge.net/rpy2/doc-dev/html/introduction.html
# TODO: Add noise classification?
# TODO: Use biosppy not mexh filtered signal, then add more features back in i.e. p wave, heights, etc.

# LATER

# TODO: Submit DRYRUN entry, entry.zip in folder is ready
# TODO: Make sure code is nice and formatted

# TODO: Deal with weird records....
    # A03509 RRvar1, RRvar2, RRvar3 NaNs
    # A03863 A03812 too
    # A00111, A00269, A00420, A00550, A00692, A01053, A01329 noisy sections
    # A00123, A00119 single inversion

"""
When submitting:
    -remove import plot from all files
    -run compress.sh, verify it included the right files, Include DRYRUN? Include saved Model?
    -make sure setup.sh includes all the right libs
    -make sure dependencies.txt has the right packages
    -make sure entry.zip is formatted correctly
    -(empty setup.sh & add validation folder+F1_score.py temporarily) make sure the whole thing runs without errors, delete pycache/vailidation/F1_score        
"""
"""
When adding features:
    -add a new features = append() line with new feature to getFeatures()
    -add a 1 to np.zeros(n) test and trainmatrix initialization in feature_extract()
"""

"""
When testing:
    -run feature_extract() (uncomment the line below it)
    -run runModel()  (uncomment the line below it)
    -go to score.py and just run the whole file
"""


class Signal(object):
    """
    An ECG/EKG signal

    Attributes:
        name: A string representing the record name.
        sampling rate/freq: the sampling rate Hz and frequency (float)
        data : 1-dimensional array with input signal data
        RPeaks : array of R Peak indices
        RRintervals : array of RR interval lengths
        RRbins : tuple of bin percents
    """

    def __init__(self, 
                 name, 
                 data, 
                 mid_bin_range=(234.85163198115271, 276.41687146297062)
                ):
        """
        Return a Signal object whose record name is *name*,
        signal data is *data*,
        RRInterval bin range is *mid_bin_range*
        """
        self.name = name
        self.sampling_rate = 300. # 300 hz
        self.sampleFreq = 1/300

        # self.data = wave.discardNoise(data) # optimize this
        self.data = wave.filterSignal(data)
        # self.data = data

        self.RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)

        self.RRintervals = wave.interval(self.RPeaks)
        self.RRbinsN = wave.interval_bin(self.RRintervals, mid_bin_range)


def deriveBinEdges(training):
    """
    This function derives bin edges from the normal EKG signals

    Parameters
    ----------
    training : tuple
        tuple of lists of training data record names and labels, 2nd tuple from wave.getPartitionedRecords()

    Returns
    -------
    edges : tuple
        tuple of bin edge values, i.e. (230,270) to use as mid_bin_range in wave.interval_bin()
    """
    
    lower = 0
    upper = 0
    normals = []
    
    for idx, val in enumerate(training[0]):
        
        if training[1][idx] == 'A':
            normals.append(val)

    for i in normals:
        
        signal = getFeaturesHardcoded(i)

        tempMean = signal[4]
        tempStd = np.sqrt(signal[3])
        
        lower += tempMean - tempStd
        upper += tempMean + tempStd
    
    lower = lower/len(normals)
    upper = upper/len(normals)
    
    return (lower,upper)

hardcoded_features = pd.read_csv("hardcoded_features.csv")

def getFeaturesHardcoded(name):
    """
    this function extract the features from the attributes of a signal
    it uses the hardcoded csv data for each signal that we saved earlier using saveSignalFeatures()

    Parameters
    ----------
    name : String
        record name

    Returns
    -------
    features : array_like
        a feature array for the given signal

    """
    
    signal = np.asarray(hardcoded_features.loc[hardcoded_features['0'] == name])[0]
        
    return signal[2:]


def getFeatures(sig):
    """
    this function extract the features from the attributes of a signal

    Parameters
    ----------
    sig : Signal object
        instantiated signal object of class Signal

    Returns
    -------
    features : list (not numpy array)
        a feature list for the given signal

    """
    
    features = [sig.name] # Record name +1 = 1
    
    features += list(sig.RRbinsN) # RR bins normal bounds +3 = 4
    features.append(np.var(sig.RRintervals)) # RR interval variance +1 = 5
    features.append(np.mean(sig.RRintervals)) # RR interval mean +1 = 6
    features.append(wave.calculate_residuals(sig.data)) # Residuals +1 = 7
    
    wtcoeff = pywt.wavedecn(sig.data, 'sym5', level=5, mode='constant')
    wtstats = wave.stats_feat(wtcoeff)
    features += wtstats.tolist() # Wavelet coefficient stats  +42 = 49
        #number of features getFeaturesHardcoded() will be returning^^, -1 for sig.name
    
    return features

def saveSignalFeatures():
    """
    This function saves all the features for each signal into a giant dataframe
    This is so we don't have to re-derive the peaks, intervals, etc. for each signal

    Parameters
    ----------
    None

    Returns
    -------
    Saves dataframe as hardcoded_features.csv where each row is a filtered signal with the following features:
        RRbins 3
        RRintervals variance 1
        RRintervals mean 1
        Residuals 1
        Wavelet coeff 42
    """
    
    records = wave.getRecords('All')[0]
    returnMatrix = []
    
    for i in records:
        sig = Signal(i, wave.load(i))
        
        features = getFeatures(sig)
        
        returnMatrix.append(features)
        
    df = pd.DataFrame(returnMatrix)
    df.to_csv('hardcoded_features.csv')

#saveSignalFeatures()

def feature_extract():
    """
    this function creates a feature matrix from partitioned data

    Parameters
    ----------
        None

    Returns
    -------
        A pickle dump of the following:
            tuple of tuples:
            test (1/10th of data) tuple:
                testing subset feature matrix, 2D array
                list of record labels N O A ~
                list of record names
            training (9/10th of data) tuple:
                training subset feature matrix, 2D array
                list of record labels N O A ~ 

    """

    records_labels = wave.getRecords('All')
    partitioned = wave.getPartitionedRecords(0) # partition nth 10th
    testing = partitioned[0]
    training = partitioned[1]

    testMatrix = np.array([np.zeros(48)])
    trainMatrix = np.array([np.zeros(48)])

    for i in records_labels[0]:
        if i in testing[0]:
            testMatrix = np.concatenate((testMatrix, [getFeaturesHardcoded(i)]))
        elif i in training[0]:
            trainMatrix = np.concatenate((trainMatrix, [getFeaturesHardcoded(i)]))
            
    testMatrix = np.delete(testMatrix, (0), axis=0) # get rid of zeros array we started with
    trainMatrix = np.delete(trainMatrix, (0), axis=0)
    
    featureMatrix = ((testMatrix, testing[1], testing[0]), (trainMatrix, training[1]))
    
    pickle.dump(featureMatrix, open("feature_matrices", 'wb'))
    
#feature_extract()

def runModel():
    """
    runs an machine learning model on our training_data with features bin1, bin2, bin3, and variance

    Parameters
    ----------
    None    
    
    Returns
    -------
        A pickle dump of the trained machine learning model (svm) and pca model

    """
    
    featureMatrix = pickle.load(open("feature_matrices", 'rb'))
        
    # Split data in train and test data
    data_test  = featureMatrix[0][0]
    answer_test  = np.asarray(featureMatrix[0][1])
    
    data_train = featureMatrix[1][0]
    answer_train = np.asarray(featureMatrix[1][1])
    
    # Creating a PCA model
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(data_train)
    total = [0,0]
    for i in np.nditer(pca.explained_variance_ratio_):
        total[0] += i
        total[1] += 1
        if total[0] > 0.9:
            pca = PCA(n_components=total[1])
            break
    pca.fit(data_train)
    data_train = pca.transform(data_train)    
    
    
    # Create and fit a svm classifier
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(data_train, answer_train)
    data_test = pca.transform(data_test)
    print(np.sum(clf.predict(data_test) == answer_test))
    
    # Create and fit a nearest-neighbor classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(data_train, answer_train) 
    KNeighborsClassifier(algorithm='auto',n_neighbors=5,weights='uniform') # try different n_neighbors
    print(np.sum(knn.predict(data_test) == answer_test))
    
    # Save the model you want to use
    pickle.dump(clf, open("model", 'wb'))
    pickle.dump(pca, open("pca", 'wb'))

#runModel()

def get_answer(record, data):
    
    sig = Signal(record, data)
    
    loaded_model = pickle.load(open("model", 'rb'))
    loaded_pca = pickle.load(open("pca", 'rb'))
    features = loaded_pca.transform([getFeatures(sig)[1:]])
    result = loaded_model.predict(features)    
    
    return result[0]
