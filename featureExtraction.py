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

# TODO: Detecting Q and S, put in a function

records = wave.getRecords('A') # N O A ~
data = wave.load(records[0])

level = 6
omission = ([1,6], True)
rebuilt = wave.decomp(data, 'sym5', level, omissions=omission)
wave.plot(rebuilt, records[0], "Index n * 0.003")

points = wave.getRPeaks(data, 150)
plt.plot(data)
plt.plot(*zip(*points), marker='o', color='r', ls='')
plt.title(records[0])
plt.show()

for i in range(0,20):
    plotData = rebuilt[points[i][0]-50:points[i][0]+50]
    qrs = wave.detect_peaks(plotData, valley=True, show=True)

# TODO: Detecting P and T waves, start using wavelets
"""
import matplotlib.pyplot as plt
records = wave.getRecords('A') # N O A ~
data = wave.load(records[0])
points = wave.getRPeaks(data, 150)
plt.plot(data)
plt.plot(*zip(*points), marker='o', color='r', ls='')
plt.title(records[0])
plt.show()

# Grabbing p and t waves
for i in range(0,20):
    plotData = data[points[i][0] + 2:points[i+1][0] - 2]
    p_or_t = wave.detect_peaks(plotData, mpd=80, show=True)
    # plt.plot(plotData)
    # plt.title(records[0])
    # plt.show()
"""
"""
for i in range(60,80):
    data = wave.load(records[i])
    points = wave.getRPeaks(data, 150)
    plt.plot(data)
    plt.plot(*zip(*points), marker='o', color='r', ls='')
    plt.title(records[i])
    plt.show()
"""

# Detecting noise

# noise_feat_mat, residuals = wave.noise_feature_extract('RECORDS')