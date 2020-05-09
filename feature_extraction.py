# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from abc import ABC, abstractmethod

class FeatureExtractors(ABC):
    @staticmethod
    def extractor_words_counts_1(dataset_bow, ngram_range = (1,1)):
        # Creating the Bag of Words model
        cv = CountVectorizer(ngram_range=ngram_range)
        X = cv.fit_transform(dataset_bow.values).toarray()
        return X

    @staticmethod
    def extractor_tf_1(dataset_bow, ngram_range):
        return FeatureExtractors.extractor_tfidf_1(dataset_bow, ngram_range, False)

    @staticmethod
    def extractor_tfidf_1(dataset_bow, ngram_range, enable_idf = True):
        tfidf_transformer = TfidfTransformer(use_idf = enable_idf)
        X_tfidf = tfidf_transformer.fit_transform(FeatureExtractors.extractor_words_counts_1(dataset_bow, ngram_range))
        return X_tfidf
