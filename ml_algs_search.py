#ODC - OptimalDisjunctiveCombination

#группа классов ищет наилучшую комбинацию алгоритмов ML.
#для поиска используется кастомная кросс-валидация и варианты объединения детектов

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from itertools import chain, combinations

from logs import *
from generic import *

class AlgsBestCombinationSearcher(object):
    def __init__(self):
        self.algs_combinations = None #[ (alg_i_name, alg_i_obj) ]
        self.folds = []
        self.combinations_quality_metrics = [] #совпадает по индексам с combinations[]

    @staticmethod
    def export_searcher_results(results, log_obj): #предполагается, что ODC и OCC имеют одинаковый вид результатов
        digits_formatter = lambda x : '{:1.3f}'.format(x)
        for algs_combi_name, q_metrics in results:
            log_obj.info('---' + algs_combi_name)
            log_obj.info(
                [metric_name + '=' + digits_formatter(metric_val) for metric_name,metric_val in q_metrics.items()])

    def prepare(self, X, y, k_folds, algs, combination_length = 4): 
        #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        def generate_algs_combinations():
            def make_all_subsets(iterable):
                list_ = list(iterable)
                return list(chain.from_iterable(combinations(list_,k) for k in range(1,self.combination_length+1)))

            self.algs_combinations = make_all_subsets(self.algs)
            #print(self.algs_combinations)                

        self.k = k_folds
        self.X = X
        self.y = y
        self.combination_length = combination_length #данный параметр нужен, если алгоритм общего вида и способен расставлять
        #алгоритмы по k местам, тогда данный параметр нужно иницииализировать через prepare
        self.algs = list(algs.items())
        generate_algs_combinations()
        self.folds = DatasetInstruments.make_stratified_split_on_stratified_k_folds(X,y,self.k)
        
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
        #метод сохраняет всю информацию о процессе в лог, а возвращает лишь ТОП комбинаций алгоритмов, отсортированных по убыванию качества
        #как происходит поиск комбинации: сначала каждый алгоритм по - одиночке проходит по фолдам, потом эти результаты комбинируются,
        #это куда эффективнее, чем подход с постоянным обучаением одних и тех же алгоритмов, которые встречаются в разных комбинациях
        def calc_quality_metrics(y_pred, y_test):
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            return {'f1': f1, 'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec, 'pred_time': -1, 'train_time': -1}

        def calc_summary_values_of_q_metrics_for_algs_combi(algs_combi_q_metrics_values_on_folds):
            dict_ = { 'f1': 0, 'auc': 0, 'acc': 0, 'prec': 0, 'rec': 0, 'pred_time': 0, 'train_time': 0 }
            n = len(algs_combi_q_metrics_values_on_folds)
            for alg_q_metrics in algs_combi_q_metrics_values_on_folds:
                new_values = alg_q_metrics.values()
                #print(new_values)
                dict_ = {key:round(((value+new_val)/n),4) for (key, value),new_val in zip(dict_.items(), new_values)} #среднее значение каждой метрики
                #print(dict_)
            return dict_

        def sort_algs_combis_by_q_metrics(keys_lambda = lambda el: el[1]['f1']):
            #пример элемента словаря ('ComplementNB', {'f1': 0.977, 'acc': 0.989, 'prec': 0.954, 'rec': 1.0})
            sorted_ = sorted(algs_combis_with_q_metrics.items(), key= keys_lambda, reverse=True)
            #return collections.OrderedDict(sorted_)
            return sorted_

        def make_single_algs_y_preds_on_folds(): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
            dict_ = {}
            for alg_name,alg_obj in self.algs:
                dict_[alg_name] = []
                for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.folds:
                    y_pred_alg = alg_obj.learn_predict(X_train = X_trainFolds, X_test = X_validFold, 
						                        y_train = y_trainFolds)
                    dict_[alg_name].append(y_pred_alg)
            return dict_

        def calc_combis_quality_metrics():
            for combi in self.algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них #Раскомментировать для логирования
                #LogsFileProvider().ml_research_general.info('---------' + str(self.get_algs_combination_name(combi)))
                for (i,(_, _, X_validFold, y_validFold)) in enumerate(self.folds):
                    algs_combi_q_metrics_values_on_folds = [] #список dict-ов с метриками
                    y_pred_combination = np.zeros(y_validFold.shape, dtype=bool)
                    for alg_name,_ in combi:
                        y_pred_alg = single_algs_y_preds[alg_name][i]
                        y_pred_combination = np.logical_or (y_pred_combination, y_pred_alg)
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().logger_ml_processing.info('y_pred_combination before' + str(dict(zip(classes.tolist(), classes_counts))))
                        #y_pred_combination = np.logical_or(y_pred_combination, y_pred_alg)
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().logger_ml_processing.info('y_pred_combination after' + str(dict(zip(classes.tolist(), classes_counts))))
                    algs_combi_q_metrics_values_on_folds.append(calc_quality_metrics(y_pred_combination, y_validFold))
	            #print('folds_shape:', X_trainFolds.shape, X_validFold.shape)
                algs_combi_mean_q_metrics = calc_summary_values_of_q_metrics_for_algs_combi(algs_combi_q_metrics_values_on_folds) 
                #LogsFileProvider().ml_research_general.info(algs_combi_mean_q_metrics) #Раскомментировать для логирования
                self.combinations_quality_metrics.append(algs_combi_mean_q_metrics)

        single_algs_y_preds = make_single_algs_y_preds_on_folds() #dict {alg_name:[y_pred_fold_i]}
        print('////////////// make_single_algs_y_preds_on_folds() done')
        calc_combis_quality_metrics()
        print('////////////// calc_combis_quality_metrics() done')

        algs_combis_with_q_metrics = dict(zip(self.get_algs_combinations_names(), self.combinations_quality_metrics))
        sorted_by_f1_results = sort_algs_combis_by_q_metrics(lambda el: (el[1]['f1'], el[1]['rec'], el[1]['pred_time']))
        AlgsBestCombinationSearcher.export_searcher_results(sorted_by_f1_results, LogsFileProvider().ml_research_combis_sorted_f1)
        sorted_by_recall_results = sort_algs_combis_by_q_metrics(lambda el:  (el[1]['rec'], el[1]['f1'], el[1]['pred_time']))
        AlgsBestCombinationSearcher.export_searcher_results(sorted_by_recall_results, LogsFileProvider().ml_research_combis_sorted_recall)
        print('////////////// logs done')

