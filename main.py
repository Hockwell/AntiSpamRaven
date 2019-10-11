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
X = FeatureExtractorsBasedOnCorpus(corpus).extractor_1() #corpus -> X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
alg = MultinomialNBAlg()
y_pred = alg.learn_predict(X_train, X_test, y_train, y_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
'''
Confusion Matrix
array([[863,  11],
       [  1, 264]])
'''
#this function computes subset accuracy
accuracy_score(y_test, y_pred) #0.9894644424934153
accuracy_score(y_test, y_pred,normalize=False) #1129 out of 1139

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)#array([ 0.98903509,  0.98903509,  0.99122807,  0.98026316,  0.98245614,0.98903509,  0.98901099,  0.99340659,  0.99340659,  0.98681319])
accuracies.mean()#0.9888085218938609
accuracies.std()#0.004090356321646494

