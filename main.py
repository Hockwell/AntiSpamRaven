#%%
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

from datasets_preprocessing import *
from algs import *
from feature_extraction import *

corpus, y = Kagle2017DatasetPreprocessors().preprocessor_1()
print('//////////////////////////// preprocessing done')
X = FeatureExtractorsBasedOnCorpus(corpus).extractor_1() #corpus -> X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print('//////////////////////////// feature extraction done')
#alg = MultinomialNBAlg()
alg = ComplementNBAlg()
y_pred = alg.learn_predict(X_train, X_test, y_train, y_test)
print('//////////////////////////// learning and prediction done')

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
'''
Confusion Matrix
array([[863,  11],
       [  1, 264]])
'''
#this function computes subset accuracy
accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred, normalize=False)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = alg.clf, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

