from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

from operator import attrgetter
from itertools import chain, combinations
from abc import ABC

import os

class ServiceData(ABC):
    PROGRAM_DIR = os.path.dirname(os.path.realpath(__file__))

class DatasetInstruments(ABC):
    @staticmethod
    def convert_dataset_from_pandas_to_numpy(X,y):
        if type(X) is (pd.Series or pd.DataFrame):
            X = X.values
        if type(y) is (pd.Series or pd.DataFrame):
            y = y.values
        return X,y

    @staticmethod
    def split_on_k_folds(X,y,k=10): #кол-во фолдов совпадает с кол-вом итераций разбиения
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
            return np.append(X_part1, X_part2, axis=0), np.append(y_part1, y_part2,axis=0)
        folds = []
        X_trainFolds = []
        X_validFold = []
        y_trainFolds = []
        y_validFold = []
        samples_amount = X.shape[0]
        valid_fold_size = int(samples_amount/k) #округление вниз
        train_folds_size = samples_amount - valid_fold_size
        #первая часть достаётся валидационному фолду, а остальные обучающим
        for i in range(k):
            lower_bound_valid = i*valid_fold_size
            upper_bound_valid = lower_bound_valid + valid_fold_size
            X_validFold, y_validFold = X[lower_bound_valid:upper_bound_valid], y[lower_bound_valid:upper_bound_valid]
            X_trainFolds, y_trainFolds = take_train_folds()
            folds.append((X_trainFolds, y_trainFolds, X_validFold, y_validFold))

        return folds

    @staticmethod
    def make_stratified_split_on_stratified_k_folds(X,y,k=10): #X,y - by numpy only
        #кол-во фолдов совпадает с кол-вом итераций разбиения
        #размер тестового фолда- 1/k от всего датасета. 
        X,y =  DatasetInstruments.convert_dataset_from_pandas_to_numpy(X,y)
        folds = []
        skf = StratifiedKFold(n_splits=k)
        for train_indices, valid_indices in skf.split(X, y):
            X_trainFolds = X[train_indices]
            X_validFold = X[valid_indices]
            y_trainFolds = y[train_indices]
            y_validFold = y[valid_indices]
            folds.append((X_trainFolds, y_trainFolds, X_validFold, y_validFold))
        return folds

    @staticmethod
    def make_shuffle_stratified_split_on_folds(X,y, test_size = 0.2, n_splits=10): #X,y - by numpy only
        #кол-во фолдов (k) НЕ совпадает с кол-вом итераций разбиения (n_splits),потому что алг. с перемешиванием (рандомизированный).
        #для задания размера фолда используется test_size
        X,y =  DatasetInstruments.convert_dataset_from_pandas_to_numpy(X,y)

        folds = []
        sss = StratifiedShuffleSplit(test_size = test_size, n_splits=n_splits)
        for train_indeces, valid_indeces in sss.split(X, y):
            X_trainFolds, X_validFold = X[train_indeces], X[valid_indeces]
            y_trainFolds, y_validFold = y[train_indeces], y[valid_indeces]
            folds.append((X_trainFolds, y_trainFolds, X_validFold, y_validFold))
        return folds
    @staticmethod
    def calc_classes_ratio(y):
        return y.value_counts(normalize=True)
        
class CollectionsInstruments(ABC):
    @staticmethod
    def sum_vals_of_similar_dicts(dicts, decimal_places): #предполагается, что все словари имеют одинаковые ключи
        keys = dicts[0].keys()
        dicts_sum_vals =  np.around(np.sum([list(dict.values()) for dict in dicts], axis=0), decimal_places)
        return dict(zip(keys, dicts_sum_vals))

    @staticmethod
    def merge_dicts(dict1, dict2):
        return {**dict1, **dict2}

    @staticmethod
    def round_all_dict_vals(dict_, decimal_places):
        return dict(zip(dict_.keys(), list(np.around(list(dict_.values()), decimal_places))))

    @staticmethod
    def round_dict_vals(dict_, decimal_places, keys): #in-place
        for key in keys:
            dict_[key] = round(dict_[key], decimal_places)

    @staticmethod
    def create_dict_by_keys_and_vals(keys, values):
        return dict(zip(keys, values))

    @staticmethod
    def delete_dict_elements_by_removal_list(dict_, keys_removal_list): #in-place
        for key in keys_removal_list:
            del dict_[key]
class MathInstruments(ABC):
    @staticmethod
    def make_subsets(iterable, max_size):
        #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        #генерим неповторяющиеся подмножества размерами от 1 до max_length
        list_ = list(iterable)
        return list(chain.from_iterable(combinations(list_,k_i) for k_i in range(1,max_size + 1))) 