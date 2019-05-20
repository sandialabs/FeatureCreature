from __future__ import print_function
from sklearn.metrics import *
import numpy as np


def verbose_print(verbose, line):
    if verbose:
        print(line)




class Analysis(object):
    def __init__(self, true, pred, prob, seed=False, verbose=True):
        print(accuracy_score(np.asarray(true), np.asarray(pred)))
        print(classification_report(np.asarray(true), np.asarray(pred)))
        print(confusion_matrix(np.asarray(true), np.asarray(pred)))
        print(roc_auc_score(np.asarray(true), np.asarray(pred)))
        """
        metrics.brier_score_loss(true, prob)
        metrics.classification_report(true, pred)
        metrics.confusion_matrix(true, pred)
        metrics.f1_score(true, pred)
        metrics.hamming_loss(true, pred)
        metrics.jaccard_similarity_score(true, pred)
        metrics.log_loss(true, pred)
        metrics.matthews_corrcoef(true, pred)
        metrics.precision_recall_curve(true, prob)
        metrics.precision_score(true, pred)
        metrics.recall_score(true, pred)
        metrics.roc_auc_score(true, prob)
        metrics.zero_one_loss(true, pred)
        """