import numpy as np
import wave # this is the wave.py file in the local folder
# np.set_printoptions(threshold=np.nan)


# Reading in matlab data

records = wave.getRecords('N')
mat = wave.load(records[0])
data = mat[:1000]

# Run Wavelet transforms

omission = ([4,5,6], True)
wave.plot(data[:600], "Original Signal", "Index n * 0.003")
rebuilt = wave.decomp(data, 'sym4', 6, omissions=omission)
wave.plot(rebuilt[:600], omission, "Index n * 0.003")

# Imperatively grabbing features

# Detecting R Peaks
xMax = np.argmax(rebuilt) # location of max peak
threshold = data[xMax] * 0.35
peaks = np.zeros_like(data)
# TODO: Find all the peak intervals using the threshold ad set them into peaks


# Detecting noise