import wave # this is the wave.py file in the local folder
import matplotlib.pyplot as plt
import featureExtraction 
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing
import challenge

records = wave.getRecords('N') # N O A ~
data = wave.load(records[7])
sig = featureExtraction.Signal(records[7],data)

feature_matrix = challenge.feature_extract(sig)