#! /bin/bash
#
# file: setup.sh
#
# This bash script performs any setup necessary in order to test your
# entry.  It is run only once, before running any other code belonging
# to your entry.

set -e
set -o pipefail

pip install --user PyWavelets-0.5.2.tar.gz
pip install --user numpy-1.12.1.tar.gz
pip install --user scipy-0.19.0.tar.gz
pip install --user pandas-0.19.2.tar.gz
pip install --user biosppy-0.1.2.zip
pip install --user rpy2-2.8.5.tar.gz
pip install --user scikit-learn-0.18.1.tar.gz

chmod a+x challenge.py
