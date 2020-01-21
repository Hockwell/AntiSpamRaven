from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from operator import attrgetter

class DatasetInstruments():
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
    def make_stratified_split_on_stratified_k_folds(X,y,k=10): #кол-во фолдов совпадает с кол-вом итераций разбиения
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
    def make_shuffle_stratified_split_on_k_folds(X,y, test_size = 0.2, n_splits=10): 
        #кол-во фолдов (k) НЕ совпадает с кол-вом итераций разбиения,потому что алг. с перемешиванием.
        #для задания размера фолда используется test_size
        folds = []
        sss = StratifiedShuffleSplit(test_size = test_size, n_splits=n_splits)
        for train_indeces, valid_indeces in sss.split(X, y):
            X_trainFolds, X_validFold = X[train_indeces], X[valid_indeces]
            y_trainFolds, y_validFold = y[train_indeces], y[valid_indeces]
            folds.append((X_trainFolds, y_trainFolds, X_validFold, y_validFold))
        return folds
            
class CollectionsInstruments():
    @staticmethod
    def sum_vals_of_similar_dicts(dicts): #предполагается, что все словари имеют одинаковые ключи
        keys = dicts[0].keys()
        dicts_vals_sum = np.sum([list(dict.values()) for dict in dicts], axis=0)
        return dict(zip(keys, dicts_vals_sum))