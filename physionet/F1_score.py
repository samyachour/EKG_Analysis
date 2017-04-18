#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

types = {'N': 0, 'A': 1, 'O': 2, '~': 3}


def read_labels(path):
    labels = []
    with open(path, 'r') as reader:
        for line in reader:
            label = types[line.split(',')[1][0]]
            labels.append(label)

    return np.array(labels)


def matlab_f1_score(ref, ans):
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
    F1p = 2 * AA[3, 3] / (sum(AA[3, :]) + sum(AA[:, 3]))
    F1 = (F1n + F1a + F1o + F1p) / 4
    print(AA)
    print('F1 measure for Normal rhythm: ' '%1.4f' % F1n)
    print('F1 measure for AF rhythm: ' '%1.4f' % F1a)
    print('F1 measure for Other rhythm: ' '%1.4f' % F1o)
    print('F1 measure for Noisy recordings: ' '%1.4f' % F1p)
    print('Final F1 measure: ' '%1.4f' % F1)


if __name__ == '__main__':

    reference = read_labels('validation/REFERENCE.csv')
    predictions = read_labels('answers.txt')
    matlab_f1_score(reference, predictions)

    from sklearn.metrics import precision_recall_fscore_support, f1_score

    print('f1 score sklearn from f1_score(): ' + str(np.mean(f1_score(reference, predictions, average='macro'))))
    print('f1 score sklearn: ' + str(np.mean(precision_recall_fscore_support(reference, predictions)[2])))
