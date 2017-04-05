import wave # this is the wave.py file in the local folder
from signal import Signal
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

# Testing P wave detection

records = wave.getRecords('N') # N O A ~
data = wave.load(records[0])
sig = Signal(records[0],data)



# TODO: Detecting Q and S, put in a function
"""
records = wave.getRecords('N') # N O A ~
data = wave.load(records[0])

level = 6
omission = ([1,6], True)
rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
wave.plot(rebuilt, records[0], "Index n * 0.003")

import matplotlib.pyplot as plt
points = wave.getRPeaks(data, 150)
plt.plot(data)
plt.plot(*zip(*points), marker='o', color='r', ls='')
plt.title(records[0])
plt.show()

for i in range(20,40):
    plotData = rebuilt
    left_limit = points[i][0]-50
    right_limit = points[i][0]+50
    if right_limit > plotData.size:
        break
    plotData = plotData[left_limit:right_limit]
    qrs = wave.detect_peaks(plotData, valley=True, show=True)
"""


# Detecting noise

# noise_feat_mat, residuals = wave.noise_feature_extract('RECORDS')