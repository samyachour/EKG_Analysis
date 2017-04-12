import model
import scipy
import numpy as np


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
