#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from datasets_preprocessing import *
from algs import *
from feature_extraction import *
from ml_algs_search import *

digits_formatter = lambda x : '{:1.3f}'.format(x)

def set_libs_settings():
    ax = sns.set(style="darkgrid")

def run_single_algs_test():
    for alg in algs.items():
        y_pred = alg[1].learn_predict(X_train, X_test, y_train)
        print(alg[0],': ', accuracy_score(y_test, y_pred))

def print_searcher_results(results): #предполагается, что ODC и OCC имеют одинаковый вид результатов
    for algs_combi_name, q_metrics in results:
        print('---',algs_combi_name)
        print([metric_name + '=' + digits_formatter(metric_val) for metric_name,metric_val in q_metrics.items()])

def run_algs_best_combination_searcher(algs):
    algs_searcher = AlgsBestCombinationSearcher()
    algs_searcher.prepare(X_train, y_train, 10, algs)
    ODC_results = algs_searcher.run_ODCSearcher()
    print_searcher_results(ODC_results)
    #OCC_results = algs_searcher.run_OCCSearcher()
    #print(OCC_results)
    print('//////////////////////////// algs search done')
       
def show_graphs_about_dataset(y):
    sns.countplot(y=y)

set_libs_settings()
corpus, y = Kagle2017DatasetPreprocessors().preprocessor_1()
X = FeatureExtractorsBasedOnCorpus(corpus).extractor_1() #corpus -> X
show_graphs_about_dataset(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


#print('//////////////////////////// learning and prediction done')

#search of best algs combination
algs = {
        'ComplementNB': ComplementNBAlg(),
        'SGDClassifier': SGDAlg()
        #'NearestCentroid': NearestCentroidAlg(),
        #'LinearSVC': LinearSVCAlg(),
        #'PassiveAggressiveClassifier': PassiveAggressiveAlg(),
        #'RidgeClassifier': RidgeAlg(),
        #'KNeighborsClassifier': KNeighborsAlg()
        }

run_algs_best_combination_searcher(algs)

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



