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
pip3 install --user xyzzy-1.0.tar.gz
pip3 install --user xyzzy-1.0.tar.gz
pip3 install --user xyzzy-1.0.tar.gz
pip3 install --user xyzzy-1.0.tar.gz
pip3 install --user xyzzy-1.0.tar.gz
pip3 install --user xyzzy-1.0.tar.gz

chmod a+x challenge.py
