from __future__ import print_function
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def verbose_print(verbose, line):
    if verbose:
        print(line)


def get_true_and_pred_CV(estimator, X, y, n_folds, cv, weights, params):
    '''Get the true positives and predicted values'''
    ys = []
    #for train_idx, valid_idx in cv.split(X, y):
    for train_idx, valid_idx in cv:
        clf = estimator
        clf.fit(X=X[train_idx], y=y[train_idx],
                sample_weight=weights[train_idx])
        cur_pred = clf.predict(X[valid_idx])
        ys.append((y[valid_idx], cur_pred))
    return ys


def fit_and_score_CV(estimator, X, y, cv, weights, n_folds=2, **params):
    '''Fit accuracy, precision, recall and Receiver Operator
    Characteristic'''
    ys = get_true_and_pred_CV(estimator, X, y, n_folds, cv, weights, params)
    cv_acc = np.fromiter(map(lambda tp: accuracy_score(tp[0], tp[1]), ys),
                         np.float)
    cv_prec = np.fromiter(map(lambda tp: precision_score(tp[0], tp[1]), ys),
                          np.float)
    cv_recall = np.fromiter(map(lambda tp: recall_score(tp[0], tp[1]), ys),
                            np.float)
    cv_roc_auc = np.fromiter(map(lambda tp: roc_auc_score(tp[0], tp[1]), ys),
                             np.float)
    return(cv_acc, cv_prec, cv_recall, cv_roc_auc)


class Analysis(object):
    def __init__(self, model, seed=False, verbose=True):
        np.random.seed(seed=seed)
        self.n_samples = model.train.shape[0]
        self.clf = model.clf
        self.weights = model.weights
        self.y = model.predictors
        self.train = model.train
        self.model = model
        self.verbose = verbose

    def cross_validate(self, n_iter=100):
        '''Actually run the cross validation'''
        cv = StratifiedShuffleSplit(self.y, n_iter=100, test_size=0.5)
        (acc, prec, recall, roc) = fit_and_score_CV(self.clf, self.train,
                                                    self.y, cv, self.weights)
        print_line = "For " + str(n_iter) + " resamples at 50%"
        verbose_print(self.verbose, print_line)
        print_line = ("\tAccuracy: %0.4f +/- %0.4f" % (np.mean(acc),
                                                       np.std(acc) * 2))
        verbose_print(self.verbose, print_line)
        print_line = ("\tPrecision: %0.4f +/- %0.4f" % (np.mean(prec),
                                                        np.std(prec) * 2))
        verbose_print(self.verbose, print_line)
        print_line = ("\tRecall: %0.4f +/- %0.4f" % (np.mean(recall),
                                                     np.std(recall) * 2))
        verbose_print(self.verbose, print_line)
        print_line = ("\tReceiver Operator, AUC: %0.4f +/- %0.4f" %
                      (np.mean(roc),
                       np.std(roc) * 2))
        verbose_print(self.verbose, print_line)
        self.acc = np.mean(acc)
        self.prec = np.mean(prec)
        self.recall = np.mean(recall)
        self.roc = np.mean(roc)

    def feature_importances(self, feature_count=200):
        print("Features sorted by their score:")
        feats = []
        if hasattr(self.clf, 'feature_importances_'):
            feats = sorted(zip(map(lambda x: round(x, 4), self.clf.feature_importances_), self.model.features), reverse=True)
        for feat in feats:
            print_line = '\t' + str(feat[1]) + '\t' + str(feat[0])
            verbose_print(self.verbose, print_line)
        self.feats = feats

    def plot_random_ROC(self, n_iter=10, filename="ROC_multiple.eps"):
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        for i in range(n_iter):
            X_train, X_test, y_train, y_test = cross_v.train_test_split(self.train,
                                                                        self.y,
                                                                        test_size=0.5)
            self.clf.fit(X_train, y_train)
            y_pred_rf = self.clf.predict_proba(X_test)[:, 1]
            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
            plt.plot(fpr_rf, tpr_rf)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(filename, format='eps', dpi=1000)
