# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from abc import ABC, abstractmethod

class MLAlgorithm(ABC):
    @abstractmethod
    def learn_predict(self, X_train, X_test, y_train):
        pass

class MultinomialNBAlg(MLAlgorithm):
    #def __init__(self):

    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred
    
class ComplementNBAlg(MLAlgorithm):

    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = ComplementNB(alpha=1.0, class_prior=None, fit_prior=True)
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred

class NearestCentroidAlg(MLAlgorithm):
    
    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = NearestCentroid()
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred

class SGDAlg(MLAlgorithm):

    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = SGDClassifier()
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred
    
class LinearSVCAlg(MLAlgorithm):

    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = LinearSVC()
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred
    
class PassiveAggressiveAlg(MLAlgorithm):

    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = PassiveAggressiveClassifier()
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred
    
class RidgeAlg(MLAlgorithm):

    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = RidgeClassifier()
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred
    
class KNeighborsAlg(MLAlgorithm): #очень медленный

    def learn_predict(self, X_train, X_test, y_train):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = KNeighborsClassifier()
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(X_test)
        return y_pred
    
