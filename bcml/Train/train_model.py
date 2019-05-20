from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np
import warnings


class Train(object):
    """docstring for TrainModel"""
    def preprocess_model(self):
        '''This allows preprocessing using logistic regression'''
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(self.train,
                                                                    self.predictors,
                                                                    test_size=0.5)
        encode = OneHotEncoder()
        logistic = LogisticRegression()
        self.clf = RandomForestClassifier(n_estimators=1000,
                                          oob_score=True, n_jobs=-1)
        self.clf.fit(X_train, y_train)
        encode.fit(self.clf.apply(X_train))
        self.predmodel = logistic.fit(encode.transform(self.clf.apply(X_train_lr)), y_train_lr)

    def train_model(self):
        '''This is standard model training'''
        '''For RandomForestClassifier to work their must be no nan values, one
        way of handling this is to use the --impute option. This uses mean
        imputation, which is the least information Simpleimputer, imputation is done
        by feature
        '''
        if np.any(np.isnan(self.train)):
            warnings.warn('RandomForestClassifier requires no missing data,\
                           features being imputed by mean')
            X = self.train
            imp = SimpleImputer(missing_values='NaN', strategy='mean')
            imp.fit(X)
            self.train = imp.transform(X)
        #self.weights = 1. / np.asarray(self.weights)
        #softmax = 1. - (np.exp(self.weights) / np.sum(np.exp(self.weights), axis=0))
        #softplus = 1. / np.log(1.0 + np.exp(0.05 * self.weights))
        #abssoft = 1. - (self.weights / (1. + np.absolute(self.weights)))
        #print(self.weights)
        '''
        self.clf = RandomForestClassifier(n_estimators=512,
                                          oob_score=True, n_jobs=-1,
                                          class_weight="balanced")
        '''
        self.clf = RandomForestClassifier(n_estimators=1000,
                                          oob_score=True, n_jobs=-1)
        self.predmodel = self.clf.fit(X=self.train, y=self.predictors, sample_weight=self.weights)
        if self.weights:
            np.savetxt('test_weights.txt', self.weights, header='Weights', fmt="%s")
        np.savetxt('test_predictors.txt', self.predictors, header='Predictor', fmt="%s")

    def __init__(self, train):
        self.input = train.input
        self.train = train.train
        self.predictors = train.predictors
        self.predictor_values = train.predictor_values
        self.features = train.feature_names
        if hasattr(train,'weights'):
            self.weights = train.weights
        else:
            self.weights = None
