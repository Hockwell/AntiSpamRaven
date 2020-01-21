#ODC - OptimalDisjunctiveCombination
#OCC - OptimalConjuctiveCombination
#группа классов ищет наилучшую комбинацию алгоритмов ML.
#для поиска используется кастомная кросс-валидация и варианты объединения детектов

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from itertools import chain, combinations
import time

from logs import *
from generic import *

class AlgsBestCombinationSearcher(object):
    def __init__(self):
        self.__algs_combinations = None #[ [(alg_i_name, alg_i_obj),(),()...],[],[]... ]
        self.__folds = []
        self.__single_algs_y_preds = {} #самостоятельные предсказания алгоритмов на фолдах
        self.__single_algs_train_pred_times = {} #{alg_name:{train_time, pred_time}}

    @staticmethod
    def __export_searcher_results(results, log_obj): #предполагается, что ODC и OCC имеют одинаковый вид результатов
        for algs_combi_name, q_metrics in results:
            log_obj.info('---' + algs_combi_name)
            log_obj.info(
                [metric_name + '=' + str(metric_val) for metric_name,metric_val in q_metrics.items()])

    @staticmethod
    def __sort_algs_combis_by_q_metrics(algs_combis_with_q_metrics, criterias = [('f1', True)]):
        def multisort(data):
            for key, enable_reverse in reversed(criterias):
                data.sort(key=lambda el: el[1][key], reverse=enable_reverse)
            return data

        return  multisort(list(algs_combis_with_q_metrics.items()))

    @staticmethod
    def __log_results(algs_combis_with_q_metrics, results_from): #results - {algs_combi_name, {metrics}}
        def switch_loggers():
            if (results_from == 1):
                return lfp.ml_OCC_sorted_f1, lfp.ml_OCC_sorted_recall
            if (results_from == 2):
                return lfp.ml_ODC_sorted_f1, lfp.ml_ODC_sorted_recall
            if (results_from == 3):
                return lfp.ml_ODC_OCC_sorted_f1, lfp.ml_ODC_OCC_sorted_recall

        lfp = LogsFileProvider()
        f1_logger, recall_logger = switch_loggers()
        sorted_by_f1_results = AlgsBestCombinationSearcher.__sort_algs_combis_by_q_metrics(algs_combis_with_q_metrics, criterias = 
                                                                                           [('f1', True), ('rec', True), ('pred_time', False)])

        AlgsBestCombinationSearcher.__export_searcher_results(sorted_by_f1_results, f1_logger)
        sorted_by_recall_results = AlgsBestCombinationSearcher.__sort_algs_combis_by_q_metrics(algs_combis_with_q_metrics,
            criterias =  [('rec', True), ('f1', True), ('pred_time', False)])
        AlgsBestCombinationSearcher.__export_searcher_results(sorted_by_recall_results, recall_logger)
        #print('////////////// logs done')

    def __test_single_algs_on_folds(self): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
        for alg_name,alg_obj in self.__algs:
            train_times_on_folds = []
            pred_time_on_folds = []
            self.__single_algs_y_preds[alg_name] = []
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.__folds:
                #time.time() - показывает реальное время, а не время CPU, поэтому результаты не очень точные, но этой точности
                #достаточно, для реализации более точного подсчета с timeit нужно писать куда больше кода. Также, нам не важно
                #получить абсолютные достоверные значения, важно дифференцировать алгоритмы друг с другом.
                t0 = time.time()
                alg_obj.learn(X_train = X_trainFolds, y_train = y_trainFolds)
                t1 = time.time()
                y_pred_alg = alg_obj.predict(X_test = X_validFold)
                t2 = time.time()
                self.__single_algs_y_preds[alg_name].append(y_pred_alg)
                train_times_on_folds.append(t1-t0)
                pred_time_on_folds.append(t2-t1)
            self.__single_algs_train_pred_times[alg_name] = {'train_time': np.mean(train_times_on_folds), 'pred_time': np.mean(pred_time_on_folds)}
        print('////////////////// test_single_algs_on_folds() done')

    def run(self, X, y, k_folds, algs):
        self.__tune(X, y, k_folds, algs)
        self.__test_single_algs_on_folds() #dict {alg_name:[y_pred_fold_i]}

        odc_results = self.__run_ODC_OCC_searcher(run_OCC = False)
        AlgsBestCombinationSearcher.__log_results(odc_results, results_from = 2)

        occ_results = self.__run_ODC_OCC_searcher(run_OCC = True)
        AlgsBestCombinationSearcher.__log_results(occ_results, results_from = 1)

        odc_occ_results = dict(odc_results)
        odc_occ_results.update(occ_results)
        AlgsBestCombinationSearcher.__log_results(odc_occ_results, results_from = 3)

    def __tune(self, X, y, k_folds, algs, combination_length = 4, det_metrics_exported_vals_length = 4, perf_metrics_exported_vals_length = 7): 
        #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        def generate_algs_combinations():
            def make_all_subsets(iterable):
                list_ = list(iterable)
                return list(chain.from_iterable(combinations(list_,k) for k in range(1,self.__combination_length+1)))

            self.__algs_combinations = make_all_subsets(self.__algs)
            #print(self.algs_combinations)                

        self.__det_metrics_exported_vals_length = det_metrics_exported_vals_length
        self.__perf_metrics_exported_vals_length = perf_metrics_exported_vals_length
        self.__k = k_folds
        self.__X = X
        self.__y = y
        self.__combination_length = combination_length #данный параметр нужен, если алгоритм общего вида и способен расставлять
        #алгоритмы по k местам, тогда данный параметр нужно иницииализировать через prepare
        self.__algs = list(algs.items())
        generate_algs_combinations()
        self.__folds = DatasetInstruments.make_stratified_split_on_stratified_k_folds(X,y,self.__k)
        
    def __get_algs_combinations_names(self, run_OCC):
        def get_algs_combination_name(algs_combi):
            alg_names_separator = ' * ' if run_OCC else ' + '
            combi_name = ''
            for alg_name,_ in list(algs_combi):
                combi_name += alg_name + alg_names_separator
            return combi_name[0:-len(alg_names_separator)]

        combi_names = []
        for combi in self.__algs_combinations:
            combi_names.append(get_algs_combination_name(combi))
        return combi_names
    
    def __run_ODC_OCC_searcher(self, run_OCC = False):
        #метод сохраняет всю информацию о процессе в лог, а возвращает лишь ТОП комбинаций алгоритмов, отсортированных по убыванию качества
        #как происходит поиск комбинации: сначала каждый алгоритм по - одиночке проходит по фолдам, потом эти результаты комбинируются,
        #это куда эффективнее, чем подход с постоянным обучаением одних и тех же алгоритмов, которые встречаются в разных комбинациях
        def calc_combis_quality_metrics(): #их бывает 2 типа: метрики производительности и качества детектирования

            def calc_detection_quality_metrics(y_pred, y_test):
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred)
                return {'f1': f1, 'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec}

            def calc_mean_perf_q_metrics_for_algs_combi(algs_combination):
                #pred_time, train_time вычисляются для комбинации подсчетом суммы значений этих метрик для каждого алгоритма в отдельности,
                #не учитывается время, которое тратится на агрегацию прогнозов (or, and функции, например), но это и не важно
                dicts = [self.__single_algs_train_pred_times[alg_name] for alg_name,_ in algs_combination]
                #print(dicts)
                return CollectionsInstruments.sum_vals_of_similar_dicts(dicts)

            def calc_mean_q_metrics_for_algs_combi(algs_combi_det_q_metrics_values_on_folds, algs_combi_mean_perfomance_q_metrics):
                def add_perfomance_q_metrics_in_results():
                    dict_['pred_time'] = round(algs_combi_mean_perfomance_q_metrics['pred_time'], self.__perf_metrics_exported_vals_length)
                    dict_['train_time'] = round(algs_combi_mean_perfomance_q_metrics['train_time'], self.__perf_metrics_exported_vals_length)

                dict_ = { 'f1': 0, 'auc': 0, 'acc': 0, 'prec': 0, 'rec': 0}
                n = len(algs_combi_det_q_metrics_values_on_folds)
                for alg_q_metrics in algs_combi_det_q_metrics_values_on_folds:
                    new_values = alg_q_metrics.values()
                    #print(new_values)
                    dict_ = {key:round(((value+new_val)/n), self.__det_metrics_exported_vals_length) for (key, value),new_val in 
                             zip(dict_.items(), new_values)} #среднее значение каждой метрики
                    #print(dict_)
                add_perfomance_q_metrics_in_results()
                return dict_

            combis_aggregation_func = np.logical_and if run_OCC else np.logical_or
            y_pred_combi_init_func = np.ones if run_OCC else np.zeros

            for combi in self.__algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них 
                #Раскомментировать для логирования
                #LogsFileProvider().ml_research_general.info('---------' + str(self.get_algs_combination_name(combi)))
                for (i,(_, _, X_validFold, y_validFold)) in enumerate(self.__folds):
                    algs_combi_det_q_metrics_on_folds = [] #список dict-ов с метриками
                    y_pred_combination = y_pred_combi_init_func(y_validFold.shape, dtype=bool)
                    for alg_name,_ in combi:
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().ml_research_general.info('y_pred_combination before' + str(dict(zip(classes.tolist(), classes_counts))))
                        y_pred_alg = self.__single_algs_y_preds[alg_name][i]
                        y_pred_combination = combis_aggregation_func(y_pred_combination, y_pred_alg)
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().ml_research_general.info('y_pred_combination after' + str(dict(zip(classes.tolist(), classes_counts))))
                    algs_combi_det_q_metrics_on_folds.append(calc_detection_quality_metrics(y_pred_combination, y_validFold))
	            #print('folds_shape:', X_trainFolds.shape, X_validFold.shape)
                algs_combi_mean_perfomance_q_metrics = calc_mean_perf_q_metrics_for_algs_combi(combi)
                algs_combi_mean_q_metrics = calc_mean_q_metrics_for_algs_combi(algs_combi_det_q_metrics_on_folds, 
                                                                               algs_combi_mean_perfomance_q_metrics) 
                #LogsFileProvider().ml_research_general.info(algs_combi_mean_q_metrics) #Раскомментировать для логирования
                algs_combis_quality_metrics.append(algs_combi_mean_q_metrics)

        algs_combis_quality_metrics = []
        
        calc_combis_quality_metrics()
        print('////////////// calc_combis_quality_metrics() done')
        algs_combis_with_q_metrics = dict(zip(self.__get_algs_combinations_names(run_OCC), algs_combis_quality_metrics))
        print('////////////////// ODC_OCC Searcher done')
        return algs_combis_with_q_metrics

