"""
This process takes in the training dataset and outputs
a data structure that includes the name of the molecule,
the predictor, and the CAS number
Attributes:
    input_file (str): This is the training file that
    is read by the output
    Instance (class): This is a private class which
    structures each instance
    Model (class): This is a public class with the
    total structure of the set
"""

from __future__ import print_function
import numpy as np
import sys
sys.path.append('..')

from bcml.KNNImpute.knnimpute import (
    knn_impute_optimistic,
)
from sklearn.impute import SimpleImputer
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict



#_possible_features = ("binhash", "padelhash", "experimentalhash")
_possible_features = ('experimentalhash', 'binhash', 'padelhash', 'userhash')


def dictitems(dict):
    if sys.version_info[0] >= 3:
        return dict.items()
    else:
        return dict.iteritems()


def verbose_print(verbose, line):
    if verbose:
        print(line)


def _get_feature_names(compounds, feature_names=False):
    """This function handles collecting the feature names"""
    if feature_names is False:
        feature_names = {}
        for compound in compounds:
            for feature in _possible_features:
                if feature in compound.keys():
                    keys = compound[feature].keys()
                    for feat in keys:
                        feature_names[feat] = 1
                        compound[feat] = compound[feature][feat]
        return (compounds, feature_names.keys())
    else:
        for compound in compounds:
            if 'padelhash' in compound.keys():
                keys = compound['padelhash'].keys()
                for feat in keys:
                    compound[feat] = compound['padelhash'][feat]
        return (compounds, feature_names)
    


class Process(object):
    """This file reads a training file"""
    def load_testing_set(self):
        """This function takes the features and
        compounds and loads them into a numpy array
        """
        for index, value in np.ndenumerate(self.test):
            compound = self.compounds[index[0]]
            feature = list(self.feature_names)[index[1]]
            if (feature in compound.keys()) and (compound[feature] is not "")\
               and (compound[feature] != "NULL")\
               and (compound[feature] != "False"):
                self.test[index] = float(compound[feature])
            else:
                self.test[index] = np.nan

    def impute_values(self, distance=False, k=5, verbose=True, simple=False):
        """This function handles the missing values from
        the training set and estimates their value, based on
        the mean and reloads them into the training set"""
        X = self.test
        if simple:
            '''For features with a small number of features, use mean
            imputation to remove NaN values'''
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X)
            self.test = imp.transform(X).astype(float)
        else:
            verbose_print(verbose, 'Imputing using KNN strategy')            
            missing_mask = np.isnan(X)
            '''Impute uing knn optimistic'''
            impute = knn_impute_optimistic(X, missing_mask=missing_mask,
                                           distance=distance, k=k)
            X = impute.astype(float)


    '''def impute_values(self, distance=False):
        """This function handles the missing values from
        the training set and estimates their value, based on
        the mean and reloads them into the training set
        A smarter way of handling this is to impute based on the
        distance of each value from the combined training and
        test sets.
        """
        for index, value in np.ndenumerate(self.test):
            column = index[1]
            if (np.isnan(value) is True) or (np.isinf(value) is True) or (np.can_cast(value, np.float64) is False) or (np.can_cast(value, np.float32) is False):
                self.test[index] = self.impute[column]'''

    def __init__(self, testing_data, features, feature_names=False):
        """This initialization function handles the heavy
        work of loading the features and processing the
        compounds"""
        self.input = testing_data
        self.features = features
        self.columns = len(self.features)
        self.rows = len(testing_data.compound)
        self.test = np.zeros((self.rows, self.columns,), dtype=np.float64)
        compounds = []
        self.test_names = []
        self.input.compound = OrderedDict(sorted(self.input.compound.items(), key=lambda t: t[0]))
        for id, compound in dictitems(self.input.compound):
            compounds.append(compound)
            self.test_names.append(id)
        (self.compounds, self.feature_names) = _get_feature_names(compounds, feature_names)
        self.load_testing_set()
        