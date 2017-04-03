import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan) # show full arrays, dataframes, etc. when printing


# Run Wavelet transforms

"""
level = 6
omission = ([5,6], True)

wave.plot(data, "Original Signal", "Index n * 0.003")
rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
wave.plot(rebuilt, omission, "Index n * 0.003")
"""
    

# Imperatively grabbing features


# Detecting R Peaks

import matplotlib.pyplot as plt
data = wave.load('A00123')
points = wave.getRPeaks(data, 150)
plt.plot(data)
plt.plot(*zip(*points), marker='o', color='r', ls='')
plt.title('A00123')
plt.show()

records = wave.getRecords('A') # N O A ~
for i in range(60,80):
    data = wave.load(records[i])
    points = wave.getRPeaks(data, 150)
    plt.plot(data)
    plt.plot(*zip(*points), marker='o', color='r', ls='')
    plt.title(records[i])
    plt.show()


# Detecting noise

# noise_feat_mat, residuals = wave.noise_feature_extract('RECORDS')