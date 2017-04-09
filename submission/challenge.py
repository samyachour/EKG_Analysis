#!/usr/bin/python3
# Example challenge entry

import sys
import scipy.io

record = sys.argv[1]

# Read waveform samples (input is in WFDB-MAT format)
mat_data = scipy.io.loadmat(record + ".mat")
samples = mat_data['val']

# Your classification algorithm goes here...
if samples[0][0] < 0:
    answer = "N"
else:
    answer = "A"

# Write result to answers.txt
answers_file = open("answers.txt", "a")
answers_file.write("%s,%s\n" % (record, answer))
answers_file.close()
