#ODC - OptimalDisjunctiveCombination
#DC - DisjunctiveCombination
#OCC - OptimalConjuctiveCombination
#CC - ConjuctiveCombination
#combis - Combinations
#группа классов ищет наилучшую комбинацию алгоритмов ML.

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from enum import Enum
import time

from logs import *
from generic import *

class AlgsBestCombinationSearcher(object):
    class CombinationsTypes(Enum):
        DISJUNCTIVE = 1
        CONJUNCTIVE = 2

    class AlgsCombination(object):
        def __init__(self, combi_type, algs_names, algs_objs):
            self.type = combi_type
            self.algs_names = algs_names #[]
            self.algs_objs = algs_objs #[]
            self.quality_metrics = None #{}

        def get_name(self):
            combi_name = ''
            separators_for_combi_names = { AlgsBestCombinationSearcher.CombinationsTypes.DISJUNCTIVE:' + ', 
                                          AlgsBestCombinationSearcher.CombinationsTypes.CONJUNCTIVE: ' * ' }
            separator_for_name = separators_for_combi_names[self.type]
            for alg_name in self.algs_names:
                combi_name += alg_name + separator_for_name
            return combi_name[0:-len(separator_for_name)]

    def __init__(self):
        self.__algs_DC = None
        self.__algs_CC = None 
        self.__folds = []
        self.__single_algs_y_preds = {} 
        self.__single_algs_train_pred_times = {} #{alg_name:{train_time, pred_time}}

    @staticmethod
    def __export_searcher_results(algs_combis, log_obj): #предполагается, что все типы комбинаций имеют одинаковый вид результатов
        for algs_combi in algs_combis:
            log_obj.info('---' + algs_combi.get_name())
            log_obj.info(
                [metric_name + '=' + str(metric_val) for metric_name,metric_val in algs_combi.quality_metrics.items()])

    @staticmethod
    def __sort_algs_combis_by_q_metrics(algs_combis, criterias = [('f1', True)]):
        def multisort(list_):
            for key, enable_reverse in reversed(criterias):
                list_.sort(key=lambda el: el.quality_metrics[key], reverse=enable_reverse)
            return list_

        return  multisort(algs_combis)

    def __export_results(self, algs_combis, results_from): #results - {algs_combi_name, {metrics}}
        def switch_loggers():
            lfp = LogsFileProvider()
            if (results_from == 1):
                return lfp.ml_OCC_sorted_f1, lfp.ml_OCC_sorted_recall
            if (results_from == 2):
                return lfp.ml_ODC_sorted_f1, lfp.ml_ODC_sorted_recall
            if (results_from == 3):
                return lfp.ml_ODC_OCC_sorted_f1, lfp.ml_ODC_OCC_sorted_recall

        def export_sorted_by(logger, criterias_list):
            def find_first_combis_with_unique_algs_in_results(entries_amount_of_alg = 1): #entries_amount_of_alg = 1 is combis_with_unique_algs
                #first combis - начальные комбинации в отсортированном перечне, попадающие под условие
                #алгоритмы содержатся в dict-е, значением является то самое кол-во комбинаций, где алгоритм был обнаружен,
                    #это значение не может превышать entries_amount_of_alg
                    #фиксируем алгоритм, запоминаем индекс в sorted_results.
                    #по индексам формируем список фильтрованных результатов, по факту они находятся в полном списке - лишняя память не расходуется
                #Данный фильтр хорош тогда, когда комбинации отсортированы по убыванию своего качества, на выходе мы получим
                #уникальные комбинации, при этом каждый алгоритм присутствует в комбинации лучшего качества среди остальных
                #комбинаций со своим участием. Уникальная комбинация - та, где есть хотя бы один уникальный алгоритм
                def mark_algs_entries_in_combi():
                    for alg_name in algs_combi.algs_names:
                        algs_entries_in_combis[alg_name] += 1

                def is_combi_content_unique_alg():
                    for alg_name in algs_combi.algs_names:
                        if algs_entries_in_combis[alg_name] == entries_amount_of_alg:
                            return True
                    return False

                combis_with_unique_algs = []
                algs_entries_in_combis = dict(zip([alg_name for alg_name,_ in self.__algs], [0 for _ in self.__algs]))
                for algs_combi in sorted_algs_combis:
                    mark_algs_entries_in_combi()
                    if (is_combi_content_unique_alg()):
                        combis_with_unique_algs.append(algs_combi)
                return combis_with_unique_algs

            sorted_algs_combis = AlgsBestCombinationSearcher.__sort_algs_combis_by_q_metrics(algs_combis, 
                                                                                         criterias = criterias_list)
            sorted_algs_combis_with_unique_algs = find_first_combis_with_unique_algs_in_results()
            logger.info("//// sorted_results_combis_with_unique_algs ////")
            AlgsBestCombinationSearcher.__export_searcher_results(sorted_algs_combis_with_unique_algs, logger)
            logger.info("//// all results ////")
            AlgsBestCombinationSearcher.__export_searcher_results(sorted_algs_combis, logger)

        f1_logger, recall_logger = switch_loggers()
        export_sorted_by(f1_logger, [('f1', True), ('rec', True), ('pred_time', False)])
        #export_sorted_by(recall_logger, [('rec', True), ('f1', True), ('pred_time', False)]) #можно включить

    def __test_single_algs_on_folds(self): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
        single_algs_y_preds = {}
        single_algs_train_pred_times = {}

        for alg_name,alg_obj in self.__algs:
            train_times_on_folds = []
            pred_time_on_folds = []
            single_algs_y_preds[alg_name] = []
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.__folds:
                #time.time() - показывает реальное время, а не время CPU, поэтому результаты не очень точные, но этой точности
                #достаточно, для реализации более точного подсчета с timeit нужно писать куда больше кода. Также, нам не важно
                #получить абсолютные достоверные значения, важно дифференцировать алгоритмы друг с другом.
                t0 = time.time()
                alg_obj.learn(X_train = X_trainFolds, y_train = y_trainFolds)
                t1 = time.time()
                y_pred_alg = alg_obj.predict(X_test = X_validFold)
                t2 = time.time()
                single_algs_y_preds[alg_name].append(y_pred_alg)
                train_times_on_folds.append(t1-t0)
                pred_time_on_folds.append(t2-t1)
            single_algs_train_pred_times[alg_name] = {'train_time': np.mean(train_times_on_folds), 'pred_time': np.mean(pred_time_on_folds)}
        print('////////////////// test_single_algs_on_folds() done')
        return single_algs_y_preds, single_algs_train_pred_times

    def run(self, X, y, k_folds, algs, enable_OCC = False):
        def export_odc_occ_general_results():
            self.__export_results(self.__algs_DC + self.__algs_CC, results_from = 3)

        self.__tune(X, y, k_folds, algs, enable_OCC)
        single_algs_y_preds, single_algs_train_pred_times = self.__test_single_algs_on_folds()  #самостоятельные предсказания алгоритмов на фолдах

        self.__run_ODC_OCC_searcher(single_algs_y_preds, single_algs_train_pred_times, run_OCC = False)
        self.__export_results(self.__algs_DC, results_from = 2)

        if (enable_OCC):
            self.__run_ODC_OCC_searcher(single_algs_y_preds, single_algs_train_pred_times, run_OCC = True)
            self.__export_results(self.__algs_CC, results_from = 1)
            export_odc_occ_general_results()

    #det - detection, perf - perfomance
    def __tune(self, X, y, k_folds, algs, enable_OCC, combination_length = 4, 
               det_metrics_exported_vals_length = 4, perf_metrics_exported_vals_length = 7): 
        #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        def make_algs_combinations():
            algs_subsets = MathInstruments.make_subsets(self.__algs, self.__combination_length+1) 
            #[ [(alg_i_name, alg_i_obj),(),()...],[],[]... ]
            self.__algs_DC = []
            if (enable_OCC):
                self.__algs_CC = []
            for subset in algs_subsets: 
                algs_names, algs_objs = zip(*subset)            
                self.__algs_DC.append(AlgsBestCombinationSearcher.AlgsCombination(
                    AlgsBestCombinationSearcher.CombinationsTypes.DISJUNCTIVE, algs_names, algs_objs))
                if (enable_OCC):
                    self.__algs_CC.append(AlgsBestCombinationSearcher.AlgsCombination(
                        AlgsBestCombinationSearcher.CombinationsTypes.CONJUNCTIVE, algs_names, algs_objs))
        self.__det_metrics_exported_vals_length = det_metrics_exported_vals_length
        self.__perf_metrics_exported_vals_length = perf_metrics_exported_vals_length
        self.__k = k_folds
        self.__X = X
        self.__y = y
        self.__combination_length = combination_length #данный параметр нужен, если алгоритм общего вида и способен расставлять
        #алгоритмы по k местам, тогда данный параметр нужно иницииализировать через prepare
        self.__algs = list(algs.items())
        make_algs_combinations() 
        self.__folds = DatasetInstruments.make_stratified_split_on_stratified_k_folds(X,y,self.__k)
        
    def __run_ODC_OCC_searcher(self, single_algs_y_preds, single_algs_train_pred_times, run_OCC = False):
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
                dicts = [single_algs_train_pred_times[alg_name] for alg_name in algs_combination.algs_names]
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
                    dict_ = { key:round(((value+new_val)/n), self.__det_metrics_exported_vals_length) for (key, value),new_val in 
                             zip(dict_.items(), new_values) } #среднее значение каждой метрики
                    #print(dict_)
                add_perfomance_q_metrics_in_results()
                return dict_

            combis_aggregation_func = np.logical_and if run_OCC else np.logical_or
            y_pred_combi_init_func = np.ones if run_OCC else np.zeros
            algs_combinations = self.__algs_CC if run_OCC else self.__algs_DC

            for combi in algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них 
                #Раскомментировать для логирования
                #LogsFileProvider().ml_research_general.info('---------' + str(self.get_algs_combination_name(combi)))
                for (i,(_, _, X_validFold, y_validFold)) in enumerate(self.__folds):
                    algs_combi_det_q_metrics_on_folds = [] #список dict-ов с метриками
                    y_pred_combination = y_pred_combi_init_func(y_validFold.shape, dtype=bool)
                    for alg_name in combi.algs_names:
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().ml_research_general.info('y_pred_combination before' + str(dict(zip(classes.tolist(), classes_counts))))
                        y_pred_alg = single_algs_y_preds[alg_name][i]
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
                combi.quality_metrics = algs_combi_mean_q_metrics
        
        calc_combis_quality_metrics()
        print('////////////// calc_combis_quality_metrics() done')
        #algs_combis_with_q_metrics = dict(zip(self.__get_algs_combinations_names(run_OCC), algs_combis_quality_metrics))
        print('////////////////// ODC_OCC Searcher done')
        #return algs_combis_with_q_metrics
