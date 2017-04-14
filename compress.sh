#!/bin/bash
#
# file: prepare-entry.sh
#
# This script automatically compresses all the relevant files
# we need for our physionet submission into an entry.zip file.
# It also acts as a list of all the relevant files we need.

# Add tar.gz files, make sure setup.sh includes libraries,
# remove/add DRYRUN, leave out test.py, plot.py, and ipnyb
# delete import plot.py file from code

zip entry.zip challenge.py model.py wave.py answers.txt AUTHORS.txt biosppy-0.1.2.zip dependencies.txt detect_peaks.py LICENSE.txt next.sh numpy-1.12.1.tar.gz pandas-0.19.2.tar.gz prepare-entry.sh rpy2-2.8.5.tar.gz PyWavelets-0.5.2.tar.gz scipy-0.19.0.tar.gz setup.sh 
