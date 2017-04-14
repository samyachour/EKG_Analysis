#!/bin/bash
#
# file: prepare-entry.sh
#
# This script automatically compresses all the relevant files
# we need for our physionet submission into an entry.zip file.
# It also acts as a list of all the relevant files we need.

zip entry.zip challenge.py model.py wave.py
