# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

import numpy as np

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
        self._clf = SGDClassifier() #LinearSVC с градиентным спуском

class SGDAlg_AdaptiveIters(MLAlgorithm):
    def __init__(self):
        pass

    def learn(self, X_train, y_train):
        self._clf = SGDClassifier(max_iter=np.ceil(10**6 / X_train.shape[0])) #формула из рекомендаций создателей библиотеки
        self._clf.fit(X_train , y_train)

class ASGDAlg_Default(MLAlgorithm):
    def __init__(self):
        self._clf = SGDClassifier(average=True)

class LinearSVCAlg_Default(MLAlgorithm): #SVM с линейной функцией
    def __init__(self):
        self._clf = svm.LinearSVC(loss='squared_hinge')

class LinearSVCAlg_Balanced(MLAlgorithm): 
    def __init__(self):
        self._clf = svm.LinearSVC(class_weight='balanced', loss='squared_hinge')

class SVCAlg_RBF_Default(MLAlgorithm):
    def __init__(self):
        self._clf = svm.SVC(kernel='rbf', gamma='scale')

class SVCAlg_RBF_Aggr(MLAlgorithm):
    def __init__(self):
        self._clf = svm.SVC(kernel='rbf', gamma='scale', C=0.01)
    
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
        
class PerceptronAlg_Default(MLAlgorithm): # is equivalent to SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None).
    def __init__(self):
        self._clf = Perceptron()

#class DecisionTreeAlg_Default(MLAlgorithm):
#    def __init__(self):
#        self._clf = Perceptron()

#class LogisticRegrAlg_Default(MLAlgorithm):
#    def __init__(self):
#        self._clf = Perceptron()

#class AdaBoostAlg_Default(MLAlgorithm):
#    def __init__(self):
#        self._clf = Perceptron()

#class XGBoostAlg_Default(MLAlgorithm):
#    def __init__(self):
#        self._clf = Perceptron()

#class LightGBMAlg_Default(MLAlgorithm):
#    def __init__(self):
#        self._clf = Perceptron()