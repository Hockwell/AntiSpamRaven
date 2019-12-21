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

class FeatureExtractorsBasedOnCorpus:
    def __init__(self, corpus):
        self.corpus = corpus

    def extractor_words_counts_1(self):
        # Creating the Bag of Words model
        cv = CountVectorizer()
        X = cv.fit_transform(self.corpus.values).toarray()
        return X

    def extractor_tfidf_1(self):
        # Creating the Bag of Words model
        cv = CountVectorizer()
        X = cv.fit_transform(self.corpus.values).toarray()
        
        return X
