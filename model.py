import wave
import pandas as pd
import numpy as np
import pickle

# NOW

# TODO: Filtering the signal downsamples it, so maybe change the sampling frequency passed into biosspy r peak detection?
# TODO: Testing different feature selections, do PCA
# TODO: Saving signal features to make it faster

# LATER

# TODO: Submit DRYRUN entry, entry.zip in folder is ready
# TODO: code cleanup/refactoring, add unit tests
# TODO: Start using rpy2 to work with alex's code to do regression http://rpy.sourceforge.net/rpy2/doc-dev/html/introduction.html
# TODO: Add back in the p wave detection if needed, or other features

# TODO: Deal with weird records....
    # A03509 RRvar1, RRvar2, RRvar3 NaNs
    # A03863 A03812 too
    # A00111, A00269, A00420, A00550, A00692, A01053, A01329 noisy sections
    # A00123, A00119 single inversion

"""
Upon submission:
    -remove import plot from all files
    -run compress.sh, verify it included the right files, Include DRYRUN? Include saved Model?
    -make sure setup.sh includes all the right libs
    -make sure dependencies.txt has the right packages
    -make sure entry.zip is formatted correctly
    -(empty setup.sh & add validation folder+F1_score.py temporarily) make sure the whole thing runs without errors, delete pycache/vailidation/F1_score        
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

    def __init__(self, name, data, mid_bin_range=(234.85, 276.42)):
        """
        Return a Signal object whose record name is *name*,
        signal data is *data*,
        RRInterval bin range is *mid_bin_range*
        """
        self.name = name
        self.sampling_rate = 300. # 300 hz
        self.sampleFreq = 1/300

        self.data = wave.filterSignal(data)
        # self.data = wave.discardNoise(self.data) # optimize this
        # self.data = data

        self.RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)

        self.RRintervals = wave.interval(self.RPeaks)
        self.RRbins = wave.interval_bin(self.RRintervals, mid_bin_range)







def deriveBinEdges(training):
    """
    This function derives bin edges from the normal EKG signals

    Parameters
    ----------
    training : tuple
        tuple of lists of training data record names and labels, first element from wave.getPartitionedRecords()

    Returns
    -------
    edges : tuple
        tuple of bin edge values, i.e. (230,270) to use as mid_bin_range in wave.interval_bin()
    """
    
    lower = 0
    upper = 0
    normals = []
    
    for idx, val in enumerate(training[0]):
        
        if training[1][idx] == 'N':
            normals.append(val)

    for i in normals:
        
        sig = Signal(i, wave.load(i))
        # print("processing " + i)
        tempMean = np.mean(sig.RRintervals)
        tempStd = np.std(sig.RRintervals)
        
        lower += tempMean - tempStd
        upper += tempMean + tempStd
    
    lower = lower/len(normals)
    upper = upper/len(normals)
    
    return (lower,upper)


def feature_extract():
    """
    this function extract the features from the attributes of a signal

    Parameters
    ----------
        None

    Returns
    -------
        A pickle dump of the feature matrix, training records (9/10th), and testing records (1/10th)

    """

    records_labels = wave.getRecords('All')
    partitioned = wave.getPartitionedRecords(1) # partition first 10th
    testing = partitioned[0]
    training = partitioned[1]

    binEdges = deriveBinEdges(training)
    bin1 = []
    bin2 = []
    bin3 = []
    variances = []
    
    for i in records_labels[0]:
        data = wave.load(i)
        sig = Signal(i, data, mid_bin_range=binEdges)
        # print("extracting " + i)
        bin1.append(sig.RRbins[0])
        bin2.append(sig.RRbins[1])
        bin3.append(sig.RRbins[2])
        variances.append(np.var(sig.RRintervals))
    
    feature_data = pd.DataFrame({'bin 1': bin1, 'bin 2': bin2, 'bin 3': bin3, 'variance' : variances, 'record': records_labels[0], 'label': records_labels[1]})
    pickle.dump(feature_data, open("feature_data", 'wb'))
    pickle.dump(testing, open("testing_records", 'wb'))
    pickle.dump(training, open("training_records", 'wb'))
    
# feature_extract()

def runModel():
    """
    runs an machine learning model on our training_data with features bin1, bin2, bin3, and variance

    Parameters
    ----------
    None    
    
    Returns
    -------
        A pickle dump of the trained machine learning model (svm)

    """
    
    df = pickle.load(open("feature_data", 'rb'))
    testing = pickle.load(open("testing_records", 'rb'))
    training = pickle.load(open("training_records", 'rb')) 
    
    testing_df = df.loc[df['record'].isin(testing[0])]
    testing_target = np.asarray(testing[1])
    training_df = df.loc[df['record'].isin(training[0])]
    training_target = np.asarray(training[1])
    
    testing_subset = testing_df[['bin 1','bin 2','bin 3', 'variance']].copy().as_matrix()
    training_subset = training_df[['bin 1','bin 2','bin 3', 'variance']].copy().as_matrix()
    
    # Split data in train and test data
    data_test  = testing_subset
    answer_test  = testing_target
    data_train = training_subset
    answer_train = training_target
    
    # Create and fit a svm classifier
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(data_train, answer_train)
    print(np.sum(clf.predict(data_test) == answer_test))
    
    # Create and fit a nearest-neighbor classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(data_train, answer_train) 
    KNeighborsClassifier(algorithm='auto',n_neighbors=5,weights='uniform') # try different n_neighbors
    print(np.sum(knn.predict(data_test) == answer_test))
    
    # Save the model you want to use
    pickle.dump(clf, open("model", 'wb'))

# runModel()

def get_answer(record, data):
    
    sig = Signal(record, data)
    
    loaded_model = pickle.load(open("model", 'rb'))
    features = np.append(np.asarray(sig.RRbins), np.var(sig.RRintervals)) # leave out variance?
    result = loaded_model.predict([features])    
    
    return result[0]
