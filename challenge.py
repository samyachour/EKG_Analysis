import model
import scipy
import numpy as np


"""
PHYSIONET SUBMISSION CODE
"""

import sys
import scipy.io

record = sys.argv[1]
# Read waveform samples (input is in WFDB-MAT format)
mat = scipy.io.loadmat("validation/" + record + ".mat")
#samples = mat_data['val']
data = np.divide(mat['val'][0],1000) # convert to millivolts

answer = model.get_answer(record, data)

# Write result to answers.txt
answers_file = open("answers.txt", "a")
answers_file.write("%s,%s\n" % (record, answer))
answers_file.close()
