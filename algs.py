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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

class MultinomialNBAlg(object):
    #def __init__(self):

    def learn_predict(self, X_train, X_test, y_train, y_test):
        # Fitting Naive Bayes classifier to the Training set
        self.clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        self.clf.fit(X_train , y_train)
        # Predicting the Test set results
        y_pred = self.clf.predict(self.X_test)
        return y_pred
        