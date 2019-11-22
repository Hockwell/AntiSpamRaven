#ODC - OptimalDisjunctiveCombination
#OCC - OptimalConjunctivalCombination

#группа классов ищет наилучшую комбинацию алгоритмов ML.
#для поиска используется кастомная кросс-валидация и варианты объединения детектов

import numpy as np
import pandas as pd

import copy

from sklearn.metrics import confusion_matrix, accuracy_score

class AlgsBestCombinationSearcher(object):
    def __init__(self):
        self.combinations = []
        self.k_folds = []
        self.combination_length = 4 #данный параметр нужен, если алгоритм общего вида и способен расставлять
        #алгоритмы по m местам, тогда данный параметр нужно иницииализировать через prepare
        self.combinations_estimates = [] #совпадает по индексам с combinations[]
        
    def prepare(self, X, y, k_folds_amount, algs, are_combinations_full = False): #Сочетания без повторений
        #algs НЕ должен компоноваться элементами None (нет алгоритма) - последний элемент, этот класс сам сделает это
        def generate_algs_combinations(): #для 4 слоёв, можно на базе рекурсии и дерева создать общий метод для n слоёв
            
            #требуется расставить эл-ты списка на 4 места,на первый взгляд необходимо найти сочетания без повторений,но
            #есть нюанс - требуется, чтобы элемент пустоты (None) повторялся, и только он, ибо необходимо проверить
            #алгоритм и по-одиночке, и в паре и т.д.
            
            algs_amount = len(self.algs)
            for i1 in range(0,algs_amount):
                    j2 = i1+1 if i1+1 <= algs_amount-1 else algs_amount-1
                    for i2 in range(j2, algs_amount):
                        j3 = i2+1 if i2+1 <= algs_amount-1 else algs_amount-1
                        for i3 in range(j3, algs_amount):
                            j4 = i3+1 if i3+1 <= algs_amount-1 else algs_amount-1
                            for i4 in range(j4 , algs_amount):
                                self.combinations.append((i1,i2,i3,i4))
                
                            
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
        algs['NotAlg'] = None
        #список алгоритмов с None - заглушка для "нет алгоритма", она обязательна, ибо
        #данные методы работают с индексами алгоритмов - их комбинируют, а уже по этим индексам будет осуществляться
        #расстановка алгоритмов по комбинациям, у пустоты тоже должен быть свой индекс
        #[(alg_name, alg_obj)]
        self.algs = list(algs.items())
        generate_algs_combinations()
        split_dataset_on_k_folds()
        
    def get_algs_combination_name(self, combi_algs_indexes):
        combi_name = ''
        for alg_index in combi_algs_indexes:
            combi_name += self.algs[alg_index][0] + ' + '
        return combi_name[0:-3]
    
    def get_algs_combinations_names(self):
        combi_names = []
        for combi in self.combinations:
            combi_names.append(self.get_algs_combination_name(combi))
        return combi_names
    
    def run_ODCSearcher(self):
        def calc_estimate_metric(y_pred, y_test):
            return accuracy_score(y_test, y_pred)
                #accuracy_score(y_test, y_pred, normalize=False)
        
        (X_trainFolds, y_trainFolds, X_validFold, y_validFold) = tuple(zip(*self.k_folds))
        for combi in self.combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них
            #print('---------', self.get_algs_combination_name(combi))
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.k_folds:
                combination_estimates_on_folds_set = []
                y_pred_combination = np.zeros(y_validFold.shape, dtype=bool)
                for alg_index in combi:
                    alg_obj = self.algs[alg_index][1]
                    if (alg_obj == None):
                        continue
                    #print('alg #',alg_index)
                    y_pred_alg = alg_obj.learn_predict(X_train = X_trainFolds, X_test = X_validFold, 
                                          y_train = y_trainFolds)
                    #print(y_pred_alg)
                    y_pred_combination = np.logical_or (y_pred_combination, y_pred_alg)
                combination_estimates_on_folds_set.append(calc_estimate_metric(y_pred_combination, y_validFold))
                #print('folds_shape:', X_trainFolds.shape, X_validFold.shape)
            algs_combi_mean_estimate = np.mean(combination_estimates_on_folds_set)
            #print(algs_combi_mean_estimate)
            self.combinations_estimates.append(round(algs_combi_mean_estimate,3))
        return dict(zip(self.get_algs_combinations_names(), self.combinations_estimates))

    def run_OCCSearcher(self):
        #будет реализован, если понадобится
        #np.logical_and(y_pred_next_alg, y_pred_algs_before)
        pass

