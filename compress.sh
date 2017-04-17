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

zip entry.zip challenge.py model.py wave.py physionet/answers.txt physionet/AUTHORS.txt physionet/DRYRUN physionet/biosppy-0.1.2.zip physionet/dependencies.txt detect_peaks.py LICENSE.txt physionet/next.sh physionet/numpy-1.12.1.tar.gz physionet/pandas-0.19.2.tar.gz physionet/prepare-entry.sh physionet/rpy2-2.8.5.tar.gz physionet/PyWavelets-0.5.2.tar.gz physionet/scipy-0.19.0.tar.gz physionet/setup.sh 
