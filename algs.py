# -*- coding: utf-8 -*-



from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

from abc import ABC, abstractmethod

class MLAlgorithm():
    def learn(self, X_train, y_train):
        self._clf.fit(X_train , y_train)

    def predict(self, X_test):
        return self._clf.predict(X_test)

    def learn_predict(self, X_train, X_test, y_train):
        self.learn(X_train, y_train)
        return self.predict(X_test)

class MultinomialNBAlg(MLAlgorithm):
    def __init__(self):
        self._clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    
class ComplementNBAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = ComplementNB(alpha=1.0, class_prior=None, fit_prior=True)

class NearestCentroidAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = NearestCentroid()

class SGDAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = SGDClassifier()
    
class LinearSVCAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = LinearSVC()
    
class PassiveAggressiveAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = PassiveAggressiveClassifier()
    
class RidgeAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = RidgeClassifier()
    
class KNeighborsAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = KNeighborsClassifier()

class RandomForestAlg_Mod1(MLAlgorithm):
    def __init__(self):
        self._clf = RandomForestClassifier(n_estimators=200, max_depth=3)

class RandomForestAlg_Mod2(MLAlgorithm):
    def __init__(self):
        self._clf = RandomForestClassifier(n_estimators=20)

class RandomForestAlg_Mod3(MLAlgorithm):
    def __init__(self):
        self._clf = RandomForestClassifier(n_estimators=20, max_depth=3)

class RandomForestAlg_Mod4(MLAlgorithm):
    def __init__(self):
        self._clf = RandomForestClassifier(n_estimators=500)

class RandomForestAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = RandomForestClassifier(n_estimators=100)

class PerceptronAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = Perceptron()

#class DecisionTreeAlg_Default(MLAlgorithm):
#    def learn_predict(self, X_train, X_test, y_train):
#        self.clf = ///
#        self.clf.fit(X_train , y_train)
#        y_pred = self.clf.predict(X_test)
#        return y_pred

#class LogisticRegrAlg_Default(MLAlgorithm):
#    def learn_predict(self, X_train, X_test, y_train):
#        self.clf = ///
#        self.clf.fit(X_train , y_train)
#        y_pred = self.clf.predict(X_test)
#        return y_pred

#class AdaBoostAlg_Default(MLAlgorithm):
#    def learn_predict(self, X_train, X_test, y_train):
#        self.clf = ///
#        self.clf.fit(X_train , y_train)
#        y_pred = self.clf.predict(X_test)
#        return y_pred

#class CatBoostAlg_Default(MLAlgorithm):
#    def learn_predict(self, X_train, X_test, y_train):
#        self.clf = ///
#        self.clf.fit(X_train , y_train)
#        y_pred = self.clf.predict(X_test)
#        return y_pred

#class XGBoostAlg_Default(MLAlgorithm):
#    def learn_predict(self, X_train, X_test, y_train):
#        self.clf = ///
#        self.clf.fit(X_train , y_train)
#        y_pred = self.clf.predict(X_test)
#        return y_pred

#class LightGBMAlg_Default(MLAlgorithm):
#    def learn_predict(self, X_train, X_test, y_train):
#        self.clf = ///
#        self.clf.fit(X_train , y_train)
#        y_pred = self.clf.predict(X_test)
#        return y_pred