# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class FeatureExtractorsForDatasets:
    def __init__(self, corpus):
        self.dataset_corpus = corpus

    def extractor_words_counts_1(self):
        # Creating the Bag of Words model
        cv = CountVectorizer()
        X = cv.fit_transform(self.dataset_corpus.values).toarray()
        return X

    def extractor_tf_1(self):
        return self.extractor_tfidf_1(enable_idf = False)

    def extractor_tfidf_1(self, enable_idf=True):
        tfidf_transformer = TfidfTransformer(use_idf = enable_idf)
        X_tfidf = tfidf_transformer.fit_transform(self.extractor_words_counts_1())
        return X_tfidf
