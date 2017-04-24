import pickle
import scipy
import numpy as np
import model

"""
SCORING
"""

types = {'N': 0, 'A': 1, 'O': 2, '~': 3}

testing = pickle.load(open("feature_matrices", 'rb'))[0]
ans = []

for i in testing[2]:
    mat = scipy.io.loadmat("../Physionet_Challenge/training2017/" + i + ".mat")
    data = np.divide(mat['val'][0],1000) # convert to millivolts

    answer = model.get_answer(i, data)
    ans.append(types[answer])

ref = np.asarray([types[i] for i in testing[0][1]])
ans = np.asarray(ans)

assert (ref.shape[0] == ans.shape[0])

AA = np.zeros((4, 4))

for n in range(ref.shape[0]):
    rec = ref[n]

    this_answer = ans[n]

    if rec == 0:
        if this_answer == 0:
            AA[0, 0] += 1
        elif this_answer == 1:
            AA[0, 1] += 1
        elif this_answer == 2:
            AA[0, 2] += 1
        elif this_answer == 3:
            AA[0, 3] += 1

    elif rec == 1:
        if this_answer == 0:
            AA[1, 0] += 1
        elif this_answer == 1:
            AA[1, 1] += 1
        elif this_answer == 2:
            AA[1, 2] += 1
        elif this_answer == 3:
            AA[1, 3] += 1

    elif rec == 2:
        if this_answer == 0:
            AA[2, 0] += 1
        elif this_answer == 1:
            AA[2, 1] += 1
        elif this_answer == 2:
            AA[2, 2] += 1
        elif this_answer == 3:
            AA[2, 3] += 1

    elif rec == 3:
        if this_answer == 0:
            AA[3, 0] += 1
        elif this_answer == 1:
            AA[3, 1] += 1
        elif this_answer == 2:
            AA[3, 2] += 1
        elif this_answer == 3:
            AA[3, 3] += 1

F1n = 2 * AA[0, 0] / (sum(AA[0, :]) + sum(AA[:, 0]))
F1a = 2 * AA[1, 1] / (sum(AA[1, :]) + sum(AA[:, 1]))
F1o = 2 * AA[2, 2] / (sum(AA[2, :]) + sum(AA[:, 2]))
# See new formula here: https://groups.google.com/forum/#!topic/physionet-challenges/64O7nhp430Q
# F1p = 2 * AA[3, 3] / (sum(AA[3, :]) + sum(AA[:, 3]))
F1 = (F1n + F1a + F1o) / 3
print('F1 measure for Normal rhythm: ' '%1.4f' % F1n)
print('F1 measure for AF rhythm: ' '%1.4f' % F1a)
print('F1 measure for Other rhythm: ' '%1.4f' % F1o)
# print('F1 measure for Noisy recordings: ' '%1.4f' % F1p)
print('Final F1 measure: ' '%1.4f' % F1)