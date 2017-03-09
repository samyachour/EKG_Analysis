import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

EKG = pd.read_csv("../MIT-BIH_Arrhythmia/100.csv", header=None, names=["time", "volts"])

plotData = EKG[:2000]

x = np.asarray(plotData['time'])
y = np.asarray(plotData['volts'])

x_smooth = np.linspace(x.min(), x.max(), 4000)
y_smooth = spline(x, y, x_smooth)

plt.plot(x_smooth, y_smooth)
plt.show()

# Windowing algorithm i'm going to implement
# http://file.scirp.org/pdf/ABB_2014101714074768.pdf

plotData['volts'].values.max()

threshold = 0.4 * plotData['volts'].values.max()