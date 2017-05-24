import model
import wave
import R
import pandas as pd
import numpy as np
from rpy2.robjects import r, pandas2ri, numpy2ri
import rpy2.robjects

pandas2ri.activate()


R.source('EKG.R')

feat_out = pd.read_csv('REFERENCE.csv', header=None, index_col=0)

feat = pd.read_csv('hardcoded_features.csv', header=0, index_col=0)

names = feat.ix[:,0]

feat_num = feat.ix[:,1:]



#feat = feat.as_array()

#nr,nc = feat.shape

#dim = rpy2.robjects.globalenv['dim']

NULL = R.null()

list_func = R.getFunction('list')

dim = R.getFunction('h')


feat = list_func(feat_out, feat_num, NULL, NULL)

validate = R.getFunction('validate')

r = R.getFunction('r')

output = r(validate, feat, feat)

#clean_list = R.getFunction('clean.list')













