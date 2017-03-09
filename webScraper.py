# MIT-BIH Atrial Fibrilation 
# https://physionet.org/atm/afdb/04015/atr/0/e/rdsamp/csv/pd/samples.csv

# MIT-BIH Normal Sinus
# https://physionet.org/atm/nsrdb/16265/atr/0/10/rdsamp/csv/pd/samples.csv

# MIT-BIH Arhhythmia 100-124 200-234
# https://physionet.org/atm/mitdb/100/atr/0/10/rdsamp/csv/pd/samples.csv
# https://physionet.org/atm/mitdb/101/atr/0/10/rdsamp/csv/pd/samples.csv
# https://physionet.org/atm/mitdb/102/atr/0/10/rdsamp/csv/pd/samples.csv
# https://physionet.org/atm/mitdb/200/atr/0/e/rdsamp/csv/pd/samples.csv

import pandas as pd

i = 100
while i < 235:
    try:
        data = pd.read_csv('https://physionet.org/atm/mitdb/' + str(i) + '/atr/0/10/rdsamp/csv/pd/samples.csv')
    except:
        print("record " + str(i) + " not found")
    else:
        data.to_csv(str(i) + ".csv")
    
    i += 1
