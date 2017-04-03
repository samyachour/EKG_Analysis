# EKG Analysis

## Who we are
The contributors to this repository make up the intern group at [MI3](http://www.choc.org/medical-intelligence-and-innovation-institute/) (Medical Intelligence and Innovation Institute) at the Children's Hospital of Orange County. We are a mixture of undergrad and grad students at Chapman University, studying pre-med, computer science, math, and data science to name a few.

## Our Goal
We are participating in the PhysioNet 2017 Challenge: [Atrial Fibrillation Classification from a Short Single Lead ECG Recording](https://physionet.org/challenge/2017/). By September 1st, 2017 we will have developed an effective model for classifying ECG (**ECG and EKG are the same thing**) signals into 4 categories:

1. Normal
2. AF (Atrial Fibrillation)
3. Other (Arrhythmia)
4. Noisy



## To Do
#### Technical
* Extract feature matrix from signal (RR Interval, Noise level, etc.)
* Create multinomial logistic regression model for EKG Classification

#### Administrative
* Write unit tests for codebase
* Flesh out readme explanation

## Dependencies
Our codebase is all in Python (3+) and makes use of your typical data science libraries: Numpy, Scipy, Pandas, Matplotlib, and Scikit Learn.

We currently only use one atypical library, [Pywavelets](https://pywavelets.readthedocs.io/en/latest/) to do wavelet decompisition/reconstruction for feature extraction.
