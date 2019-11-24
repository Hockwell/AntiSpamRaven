
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
from ml_algs_search import *

def run_single_algs_test():
    for alg in algs.items():
        y_pred = alg[1].learn_predict(X_train, X_test, y_train)
        print(alg[0],': ', accuracy_score(y_test, y_pred))

def run_algs_best_combination_searcher(algs):
    algs_searcher = AlgsBestCombinationSearcher()
    algs_searcher.prepare(X_train, y_train, 10, algs)
    results_str = algs_searcher.run_ODCSearcher()
    #results_str2 = algs_searcher.run_OCCSearcher()
    print(results_str)
    #print(results_str2)
        
corpus, y = Kagle2017DatasetPreprocessors().preprocessor_1()
print('//////////////////////////// preprocessing done')
X = FeatureExtractorsBasedOnCorpus(corpus).extractor_1() #corpus -> X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
print('//////////////////////////// feature extraction done')

#alg =  NearestCentroidAlg()
#alg =  ComplementNBAlg()
#alg =  SGDAlg()

#y_pred = alg.learn_predict(X_train, X_test, y_train, y_test)
#print('//////////////////////////// learning and prediction done')

#search of best algs combination
#algs = {
#        'ComplementNB': ComplementNBAlg(),
#        'SGDClassifier': SGDAlg(),
#        'NearestCentroid': NearestCentroidAlg(),
#        'LinearSVC': LinearSVCAlg(),
#        'PassiveAggressiveClassifier': PassiveAggressiveAlg(),
#        'RidgeClassifier': RidgeAlg(),
#        #'KNeighborsClassifier': KNeighborsAlg(),
#        }
algs = {
        'ComplementNB': ComplementNBAlg(),
        #'SGDClassifier': SGDAlg()
        }

run_algs_best_combination_searcher(algs)
print('//////////////////////////// algs search done')

#this function computes subset accuracy
#accuracy_score(y_test, y_pred)
#accuracy_score(y_test, y_pred, normalize=False)

#Making the Confusion Matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

# Applying k-Fold Cross Validation
#accuracies = cross_val_score(estimator = alg.clf, X = X_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()



