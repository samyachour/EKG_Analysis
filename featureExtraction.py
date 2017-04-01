import numpy as np
import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan)


# Reading in matlab data

records = wave.getRecords('N')
mat = wave.load(records[5])
data = mat[:1000]

# Run Wavelet transforms

level = 6
omission = ([], False)

#wave.plot(data, "Original Signal", "Index n * 0.003")
rebuilt = wave.decomp(data, 'sym4', level, omissions=omission)
#wave.plot(rebuilt, omission, "Index n * 0.003")

# Imperatively grabbing features

# Detecting R Peaks
rrData = rebuilt.tolist()
threshold = max(rrData) * 0.35
peaks = np.zeros_like(rebuilt).tolist()
# TODO: Organize this into a function in wave.py, 
#       work on excluding r peaks that don't give good intervals, detect inversion (abs values?)

for idx, i in enumerate(rrData):
    
    if i > threshold:
        peaks[idx] = i

peakXs = []
for idx, i in enumerate(peaks):
    local_max = 0
    
    if i > local_max:
        local_max = i
    
    if idx != len(peaks) - 1:
        nextMV = peaks[idx + 1]
    if nextMV == 0 and local_max != 0:
        peakXs.append([idx-3, data[idx-3]]) # keep the - 3?
        local_max = 0

import matplotlib.pyplot as plt
plt.plot(data)
for i in peakXs:
    plt.plot(i[0], i[1], marker='o', markersize=8, color="red")
plt.ylabel("mV")
plt.xlabel("Indexes")
plt.title("R peak guesses")
plt.show()


# Detecting noise