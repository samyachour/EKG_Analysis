import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing


# Reading in matlab data

records = wave.getRecords('N')
mat = wave.load(records[9])
data = mat[:1000]

# Run Wavelet transforms

level = 6
omission = ([1,2], True)

#wave.plot(data, "Original Signal", "Index n * 0.003")
rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
#wave.plot(rebuilt, omission, "Index n * 0.003")

# Imperatively grabbing features


# Detecting R Peaks

# TODO: work on excluding r peaks that don't give good intervals, detect inversion (abs values?)
# Play with these params
peaks = wave.detect_peaks(rebuilt, data, mpd=50, mph=0.5, show=True, valley=True)

# Detecting noise


# noise_feat_mat, residuals = wave.noise_feature_extract('RECORDS')