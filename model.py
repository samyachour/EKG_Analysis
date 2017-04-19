import wave
import pandas as pd
import numpy as np
import pickle

# NOW

# TODO: Remove 10 % of N O A ~, rederive bins w/ 9/10ths, test on remaining 1/10th
# TODO: use RR interval variance as a feature, maybe add Andy's features
# TODO: get statistical significance of certain variables (p values), try PCA (rPY2), stepwise selection?

# LATER

# TODO: Submit DRYRUN entry, entry.zip in folder is ready
# TODO: code cleanup/refactoring, add unit tests
# TODO: Start using rpy2 to work with alex's code to do regression http://rpy.sourceforge.net/rpy2/doc-dev/html/introduction.html
# TODO: Add back in the p wave detection if needed

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
        # self.data = wave.discardNoise(self.data) # optimize this
        # self.data = data

        self.RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)

        self.RRintervals = wave.interval(self.RPeaks)
        self.RRbins = wave.interval_bin(self.RRintervals)


print(wave.getPartitionedRecords(0))

def feature_extract():
    """
    this function extract the features from the attributes of a signal

    Parameters
    ----------
        None

    Returns
    -------
        A dataframe with features

    """

    records = wave.getRecords('All')

    labels = records[0]
    bin1 = []
    bin2 = []
    bin3 = []
    
    for i in labels:
        data = wave.load(i)
        sig = Signal(i, data)
        bin1.append(sig.RRbins[0])
        bin2.append(sig.RRbins[1])
        bin3.append(sig.RRbins[2])
    
    training = pd.DataFrame({'bin 1': bin1, 'bin 2': bin2, 'bin 3': bin3, 'record': records[0], 'label': records[1]})
    training.to_csv('training_data.csv')

    return training

def runModel():
    """
    runs an machine learning model on our training_data with features bin1, bin2, bin3, and variance

    Parameters
    ----------
        None

    Returns
    -------
        A trained model

    """
    
    target = np.asarray(wave.getRecords('All')[1])
    df = pd.read_csv('training_data.csv')
    subset = df.loc[:,'bin 1':'bin 3'].as_matrix()
    
    # Split iris data in train and test data
    # A random permutation, to split the data randomly
    np.random.seed(0)
    indices = np.random.permutation(len(subset))
    data_train = subset[indices[:-100]]
    answer_train = target[indices[:-100]]
    data_test  = subset[indices[-100:]]
    answer_test  = target[indices[-100:]]
    
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
    pickle.dump(knn, open("model", 'wb'))



def get_answer(record, data):
    
    sig = Signal(record, data)
    
    loaded_model = pickle.load(open("model", 'rb'))
    result = loaded_model.predict([np.asarray(sig.RRbins)])    
    
    return result[0]