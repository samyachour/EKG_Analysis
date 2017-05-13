import wave
import numpy as np
import pandas as pd
import pickle
import pywt

# NOW

# TODO: submit to physionet correctly, maybe leave out numpy? Add pandas
# TODO: Andy will do this:
    #Start using rpy2 to work with alex's code to do regression http://rpy.sourceforge.net/rpy2/doc-dev/html/introduction.html
    #Or just port the r code to python, if so remove rpy2 binaries and the pip line in setup.sh

# TODO: Add noise classification?

# LATER

# TODO: Deal with weird records....
    # A03509 RRvar1, RRvar2, RRvar3 NaNs
    # A03863 A03812 too
    # A00111, A00269, A00420, A00550, A00692, A01053, A01329 noisy sections
    # A00123, A00119 single inversion

"""
When submitting (use chrome):
    -run compress.sh, verify it included the right files,
        Include DRYRUN? Include saved pickle Model?
    -remove import plot from all files,
        delete line >>>'hardcoded_features = pd.read_csv("hardcoded_features.csv")'
        comment out all code outside functions (might be runModel() and feature_extract())
    -make sure setup.sh + dependencies.txt includes all the right libs
    -make sure entry.zip is formatted correctly, move files out of physionet folder
    -comment out all 'pip install' lines setup.sh, add validation folder,
        run ./prepare-entry.sh, copy in full answers.txt
    -delete new pycache, KNN_model, & vailidation folder, undo commenting in setup.sh
    -compress and submit!
"""
"""
When adding features:
    -add a new 'features.append(newFeature)' line to getFeatures()
    -re run saveSignalFeatures() to make a new harcoded_features.csv
"""

"""
When testing:
    -run feature_extract() (uncomment the line below it)
    -run runModel()  (uncomment the line below it)
    -go to score.py and just run the whole file
"""


class Signal(object):
    
    def __init__(self,
                 name,
                 data,
                 rr_bin_range=(234.85163198115271, 276.41687146297062),
                 p_height_range=(1.4044214049249117, 1.6578494444983445),
                 pp_bin_range=(231.13977553262845, 280.31128124840563),
                 pr_bin_range=(33.895661115441065, 52.440635275728425)
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
        self.data = wave.filterSignalBios(data)
        # self.data = data

        self.RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)
        if np.mean([self.data[i] for i in self.RPeaks]) < 0: # signal is inverted
            self.data = -self.data

        self.baseline = wave.getBaseline(self)

        self.RRintervals = wave.interval(self.RPeaks)
        self.RRbinsN = wave.interval_bin(self.RRintervals, rr_bin_range)

        self.PWaves = wave.getPWaves(self)
        self.PHeights = np.asarray([self.data[i] - self.baseline for i in self.PWaves])
        minPHeight = 1.17975561806
        self.PHeights = np.add(self.PHeights, minPHeight)
        self.PHeights = np.square(self.PHeights)
        self.PHeightbinsN = wave.interval_bin(self.PHeights, p_height_range)

        self.PPintervals = wave.interval(self.PWaves)
        self.PPbinsN = wave.interval_bin(self.PPintervals, pp_bin_range)

        self.PRintervals = self.RPeaks[1:] - self.PWaves
        self.PRbinsN = wave.interval_bin(self.PRintervals, pr_bin_range)

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

        if training[1][idx] == 'N':
            normals.append(val)

    for i in normals:

        signal = getFeaturesHardcoded(i)

        tempMean = signal[56]
        tempStd = signal[57]

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

    features = [sig.name]

    features += list(sig.RRbinsN)
    features.append(np.var(sig.RRintervals))

    features.append(wave.calculate_residuals(sig.data))

    wtcoeff = pywt.wavedecn(sig.data, 'sym5', level=5, mode='constant')
    wtstats = wave.stats_feat(wtcoeff)
    features += wtstats.tolist()

    features.append(wave.diff_var(sig.RRintervals.tolist()))
    features.append(wave.diff_var(sig.RRintervals.tolist(), skip=3))

    #TODO: do cal_stats for all these features?

    features.append(np.mean(sig.PHeights))
    features.append(np.var(sig.PHeights))
    features += list(sig.PHeightbinsN)

    features.append(np.mean(sig.PPintervals))
    features.append(np.var(sig.PPintervals))
    features += list(sig.PPbinsN)

    features.append(np.mean(sig.PPintervals))
    features.append(np.var(sig.PRintervals))
    features += list(sig.PRbinsN)


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
    Saves dataframe as hardcoded_features.csv where each row is a filtered signal with the getFeatures() features
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
    partitioned = wave.getPartitionedRecords(0) # partition nth 10th, 0-9
    testing = partitioned[0]
    training = partitioned[1]

    testMatrix = np.array([getFeaturesHardcoded("A00001")])
    trainMatrix = np.array([getFeaturesHardcoded("A00001")])

    for i in records_labels[0]:
        if i in testing[0]:
            testMatrix = np.concatenate((testMatrix, [getFeaturesHardcoded(i)]))
        elif i in training[0]:
            trainMatrix = np.concatenate((trainMatrix, [getFeaturesHardcoded(i)]))

    testMatrix = np.delete(testMatrix, (0), axis=0) # get rid of A00001 initialization array
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
    try:
        features = loaded_pca.transform([getFeatures(sig)[1:]])
        result = loaded_model.predict(features)

        return result[0]
    except:
        return '~' # return noise if we get an error in detection
