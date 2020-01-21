# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from abc import ABC, abstractmethod

class FeatureExtractors(ABC):
    @staticmethod
    def extractor_words_counts_1(dataset_corpus):
        # Creating the Bag of Words model
        cv = CountVectorizer()
        X = cv.fit_transform(dataset_corpus.values).toarray()
        return X

    @staticmethod
    def extractor_tf_1(dataset_corpus):
        return FeatureExtractors.extractor_tfidf_1(dataset_corpus, enable_idf = False)

    @staticmethod
    def extractor_tfidf_1(dataset_corpus, enable_idf=True):
        tfidf_transformer = TfidfTransformer(use_idf = enable_idf)
        X_tfidf = tfidf_transformer.fit_transform(FeatureExtractors.extractor_words_counts_1(dataset_corpus))
        return X_tfidf
