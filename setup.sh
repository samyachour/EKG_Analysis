#! /bin/bash
#
# file: setup.sh
#
# This bash script performs any setup necessary in order to test your
# entry.  It is run only once, before running any other code belonging
# to your entry.

set -e
set -o pipefail

# Example: install a local package in pip format
#pip3 install --user xyzzy-1.0.tar.gz
#pip3 install --user scipy-0.19.0.tar.gz
#pip3 install --user matplotlib-2.0.0.tar.gz
#pip3 install --user PyWavelets-0.5.2.tar.gz
#pip3 install --user pandas-0.19.2.tar.gz
#pip3 install --user numpy-1.12.1.tar.gz
#pip3 install --user xyzzy-1.0.tar.gz

chmod a+x challenge.py
