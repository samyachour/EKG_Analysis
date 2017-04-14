# EKG Analysis

## Who we are
The contributors to this repository make up the intern group at [MI3](http://www.choc.org/medical-intelligence-and-innovation-institute/) (Medical Intelligence and Innovation Institute) at the Children's Hospital of Orange County. We are a mixture of undergrad and grad students at Chapman University, studying pre-med, computer science, math, and data science to name a few.

## Our Goal
We are participating in the PhysioNet 2017 Challenge: [Atrial Fibrillation Classification from a Short Single Lead ECG Recording](https://physionet.org/challenge/2017/). By September 1st, 2017 we will have developed an effective model for classifying ECG (**ECG and EKG are the same thing**) signals into 4 categories:

1. Normal
2. AF (Atrial Fibrillation)
3. Other (Arrhythmia)
4. Noisy

The files you want to look at to understand the logic of our algorithm are [wave.py](https://github.com/samyachour/EKG_Analysis/blob/master/wave.py) and [model.py](https://github.com/samyachour/EKG_Analysis/blob/master/model.py)

## Dependencies
Our codebase is all in Python (3+) and makes use of your typical data science libraries: Numpy, Scipy, Pandas, Matplotlib, and Scikit Learn.

We currently only use one atypical library, [Pywavelets](https://pywavelets.readthedocs.io/en/latest/) to do wavelet decompisition/reconstruction for feature extraction.
