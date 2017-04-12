import wave
import numpy as np
import plot
import model

# TODO: Debug record errors
# TODO: code cleanup/refactoring, add unit tests
# TODO: start calling R from python

# TODO: Optimize noisy section removal, add back in the p wave detection

# A03509 RRvar1, RRvar2, RRvar3 NaNs
# A03863 A03812 too
# A00111, A00269, A00420, A00550, A00692, A01053, A01329 noisy sections
# A00123, A00119 single inversion

class Signal(object):
    """
    An ECG/EKG signal

    Attributes:
        name: A string representing the record name.
        data : 1-dimensional array with input signal data
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

        #self.data = wave.discardNoise(data) # optimize this
        self.data = data

        self.RPeaks = wave.getRPeaks(self.data, sampling_rate=self.sampling_rate)

        self.RRintervals = wave.interval(self.RPeaks)


records = wave.getRecords('All')

for i in records:
    data = wave.load(i)
    sig = Signal(i, data)
    coords = [(i, sig.data[i]) for i in np.nditer(sig.RPeaks)]
    plot.plotCoords(sig.data, coords)











"""
PHYSIONET SUBMISSION CODE
Add tar.gz files, make sure setup.sh includes libraries, remove/add DRYRUN,
    leave out test.py, plot.py, and ipnyb
"""

import sys
import scipy.io

record = sys.argv[1]
#record = 'A00001'
# Read waveform samples (input is in WFDB-MAT format)
mat = scipy.io.loadmat("validation/" + record + ".mat")
#samples = mat_data['val']
data = np.divide(mat['val'][0],1000) # convert to millivolts

answer = model.get_answer(record, data)

# Write result to answers.txt
answers_file = open("answers.txt", "a")
answers_file.write("%s,%s\n" % (record, answer))
answers_file.close()
