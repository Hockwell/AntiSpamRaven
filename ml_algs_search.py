#ODC - OptimalDisjunctiveCombination

#группа классов ищет наилучшую комбинацию алгоритмов ML.
#для поиска используется кастомная кросс-валидация и варианты объединения детектов

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

import copy
import json 
from logs import *

from itertools import chain, combinations

class AlgsBestCombinationSearcher(object):
    def __init__(self):
        self.algs_combinations = None #[ (alg_i_name, alg_i_obj) ]
        self.k_folds = []
        self.combinations_quality_metrics = [] #совпадает по индексам с combinations[]
        
    def prepare(self, X, y, k_folds_amount, algs, combination_length = 4): #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        def generate_algs_combinations():
            def make_all_subsets(iterable):
                list_ = list(iterable)
                return list(chain.from_iterable(combinations(list_,k) for k in range(1,self.combination_length+1)))

            self.algs_combinations = make_all_subsets(self.algs)
            print(self.algs_combinations)                
        def split_dataset_on_k_folds():
            #folds: k-1 - train, k-ый - valid
            def take_train_folds():
                lower_bound_train = upper_bound_valid if upper_bound_valid < samples_amount-1 else 0
                upper_bound_train = lower_bound_train + train_folds_size
                #print(lower_bound_train, upper_bound_train)
                if (upper_bound_train > samples_amount-1):
                    upper_bound_train = upper_bound_train - (samples_amount-1)
                    X_part1, y_part1 = X[lower_bound_train: samples_amount], y[lower_bound_train: samples_amount] 
                else:
                    X_part1, y_part1 = [],[]
                    #part1 - до конца датасета, part2 -с начала датасета
                X_part2, y_part2 = X[:upper_bound_train], y[:upper_bound_train]
                #return X_part1.append(X_part2, ignore_index=True), y_part1.append(y_part2, ignore_index=True)
                return np.append(X_part1, X_part2, axis=0), np.append(y_part1, y_part2,axis=0)
                
            X_trainFolds = []
            X_validFold = []
            y_trainFolds = []
            y_validFold = []
            samples_amount = X.shape[0]
            valid_fold_size = int(samples_amount/self.k) #округление вниз
            train_folds_size = samples_amount - valid_fold_size
            #первая часть достаётся валидационному фолду, а остальные обучающим
            for i in range(self.k):
                lower_bound_valid = i*valid_fold_size
                upper_bound_valid = lower_bound_valid + valid_fold_size
                X_validFold, y_validFold = X[lower_bound_valid:upper_bound_valid], y[lower_bound_valid:upper_bound_valid]
                X_trainFolds, y_trainFolds = take_train_folds()
                self.k_folds.append((X_trainFolds, y_trainFolds, X_validFold, y_validFold))
        
        self.k = k_folds_amount
        self.X = X
        self.y = y
        self.combination_length = combination_length #данный параметр нужен, если алгоритм общего вида и способен расставлять
        #алгоритмы по k местам, тогда данный параметр нужно иницииализировать через prepare
        self.algs = list(algs.items())
        generate_algs_combinations()
        split_dataset_on_k_folds()
        
    def get_algs_combination_name(self, algs_combi):
        combi_name = ''
        for alg_name,_ in list(algs_combi):
            combi_name += alg_name + ' + '
        return combi_name[0:-3]
    
    def get_algs_combinations_names(self):
        combi_names = []
        for combi in self.algs_combinations:
            combi_names.append(self.get_algs_combination_name(combi))
        return combi_names
    
    def run_ODCSearcher(self):
        def calc_estimate_metric(y_pred, y_test):
            return accuracy_score(y_test, y_pred)
		        #accuracy_score(y_test, y_pred, normalize=False)
        (X_trainFolds, y_trainFolds, X_validFold, y_validFold) = tuple(zip(*self.k_folds))
        for combi in self.algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них #Раскомментировать для логирования
            LogsFileProvider().logger_ml_processing.info('---------' + str(self.get_algs_combination_name(combi)))
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.k_folds:
                combination_q_metrics_on_folds_set = []
                y_pred_combination = np.zeros(y_validFold.shape, dtype=bool)
                for alg_name,alg_obj in combi:
                    y_pred_alg = alg_obj.learn_predict(X_train = X_trainFolds, X_test = X_validFold, 
						                    y_train = y_trainFolds)
                    y_pred_combination = np.logical_or (y_pred_combination, y_pred_alg)
                    #Раскомментировать для логирования
                    #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                    #LogsFileProvider().logger_ml_processing.info('y_pred_combination before' + str(dict(zip(classes.tolist(), classes_counts))))
                    #y_pred_combination = np.logical_or(y_pred_combination, y_pred_alg)
                    #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                    #LogsFileProvider().logger_ml_processing.info('y_pred_combination after' + str(dict(zip(classes.tolist(), classes_counts))))

                combination_q_metrics_on_folds_set.append(calc_estimate_metric(y_pred_combination, y_validFold))
	        #print('folds_shape:', X_trainFolds.shape, X_validFold.shape)
            algs_combi_mean_q_metric = np.mean(combination_q_metrics_on_folds_set)
            LogsFileProvider().logger_ml_processing.info(algs_combi_mean_q_metric) #Раскомментировать для логирования
            self.combinations_quality_metrics.append(round(algs_combi_mean_q_metric,3))
        return dict(zip(self.get_algs_combinations_names(), self.combinations_quality_metrics))


