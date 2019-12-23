#ODC - OptimalDisjunctiveCombination
#OCC - OptimalConjuctiveCombination
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
        self.single_algs_y_preds = {} #самостоятельные предсказания алгоритмов на фолдах

    @staticmethod
    def export_searcher_results(results, log_obj): #предполагается, что ODC и OCC имеют одинаковый вид результатов
        digits_formatter = lambda x : '{:1.3f}'.format(x)
        for algs_combi_name, q_metrics in results:
            log_obj.info('---' + algs_combi_name)
            log_obj.info(
                [metric_name + '=' + digits_formatter(metric_val) for metric_name,metric_val in q_metrics.items()])

    @staticmethod
    def sort_algs_combis_by_q_metrics(algs_combis_with_q_metrics, keys_lambda = lambda el: el[1]['f1']):
        #пример элемента словаря ('ComplementNB', {'f1': 0.977, 'acc': 0.989, 'prec': 0.954, 'rec': 1.0})
        sorted_ = sorted(algs_combis_with_q_metrics.items(), key= keys_lambda, reverse=True)
        #return collections.OrderedDict(sorted_)
        return sorted_

    @staticmethod
    def log_results(algs_combis_with_q_metrics, results_from): #results - {algs_combi_name, {metrics}}
        def switch_loggers():
            if (results_from == 1):
                return lfp.ml_OCC_sorted_f1, lfp.ml_OCC_sorted_recall
            if (results_from == 2):
                return lfp.ml_ODC_sorted_f1, lfp.ml_ODC_sorted_recall
            if (results_from == 3):
                return lfp.ml_ODC_OCC_sorted_f1, lfp.ml_ODC_OCC_sorted_recall

        lfp = LogsFileProvider()
        f1_logger, recall_logger = switch_loggers()
        sorted_by_f1_results = AlgsBestCombinationSearcher.sort_algs_combis_by_q_metrics(algs_combis_with_q_metrics, 
                                                                                         lambda el: (el[1]['f1'], el[1]['rec'], el[1]['pred_time']))
        AlgsBestCombinationSearcher.export_searcher_results(sorted_by_f1_results, f1_logger)
        sorted_by_recall_results = AlgsBestCombinationSearcher.sort_algs_combis_by_q_metrics(algs_combis_with_q_metrics,
            lambda el:  (el[1]['rec'], el[1]['f1'], el[1]['pred_time']))
        AlgsBestCombinationSearcher.export_searcher_results(sorted_by_recall_results, recall_logger)
        #print('////////////// logs done')

    def make_single_algs_y_preds_on_folds(self): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
        for alg_name,alg_obj in self.algs:
            self.single_algs_y_preds[alg_name] = []
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.folds:
                y_pred_alg = alg_obj.learn_predict(X_train = X_trainFolds, X_test = X_validFold, 
						                    y_train = y_trainFolds)
                self.single_algs_y_preds[alg_name].append(y_pred_alg)
        print('////////////////// make_single_algs_y_preds_on_folds() done')

    def run(self, X, y, k_folds, algs):
        self.tune(X, y, k_folds, algs)
        self.make_single_algs_y_preds_on_folds() #dict {alg_name:[y_pred_fold_i]}

        odc_results = self.run_ODC_OCC_searcher(run_OCC = False)
        AlgsBestCombinationSearcher.log_results(odc_results, results_from = 2)

        occ_results = self.run_ODC_OCC_searcher(run_OCC = True)
        AlgsBestCombinationSearcher.log_results(occ_results, results_from = 1)

        odc_occ_results = dict(odc_results)
        odc_occ_results.update(occ_results)
        AlgsBestCombinationSearcher.log_results(odc_occ_results, results_from = 3)

    def tune(self, X, y, k_folds, algs, combination_length = 4): 
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
        
    def get_algs_combinations_names(self, run_OCC):
        def get_algs_combination_name(algs_combi):
            alg_names_separator = ' * ' if run_OCC else ' + '
            combi_name = ''
            for alg_name,_ in list(algs_combi):
                combi_name += alg_name + alg_names_separator
            return combi_name[0:-len(alg_names_separator)]

        combi_names = []
        for combi in self.algs_combinations:
            combi_names.append(get_algs_combination_name(combi))
        return combi_names
    
    def run_ODC_OCC_searcher(self, run_OCC = False):
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

        def calc_combis_quality_metrics():
            combis_aggregation_func = np.logical_and if run_OCC else np.logical_or
            y_pred_combi_init_func = np.ones if run_OCC else np.zeros

            for combi in self.algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них #Раскомментировать для логирования
                #LogsFileProvider().ml_research_general.info('---------' + str(self.get_algs_combination_name(combi)))
                for (i,(_, _, X_validFold, y_validFold)) in enumerate(self.folds):
                    algs_combi_q_metrics_values_on_folds = [] #список dict-ов с метриками
                    y_pred_combination = y_pred_combi_init_func(y_validFold.shape, dtype=bool)
                    for alg_name,_ in combi:
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().ml_research_general.info('y_pred_combination before' + str(dict(zip(classes.tolist(), classes_counts))))
                        y_pred_alg = self.single_algs_y_preds[alg_name][i]
                        y_pred_combination = combis_aggregation_func(y_pred_combination, y_pred_alg)
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().ml_research_general.info('y_pred_combination after' + str(dict(zip(classes.tolist(), classes_counts))))
                    algs_combi_q_metrics_values_on_folds.append(calc_quality_metrics(y_pred_combination, y_validFold))
	            #print('folds_shape:', X_trainFolds.shape, X_validFold.shape)
                algs_combi_q_metrics_mean = calc_summary_values_of_q_metrics_for_algs_combi(algs_combi_q_metrics_values_on_folds) 
                #LogsFileProvider().ml_research_general.info(algs_combi_mean_q_metrics) #Раскомментировать для логирования
                algs_combis_quality_metrics.append(algs_combi_q_metrics_mean)

        algs_combis_quality_metrics = []
        
        calc_combis_quality_metrics()
        print('////////////// calc_combis_quality_metrics() done')
        algs_combis_with_q_metrics = dict(zip(self.get_algs_combinations_names(run_OCC), algs_combis_quality_metrics))
        print('////////////////// ODC_OCC Searcher done')
        #print(algs_combis_with_q_metrics)
        return algs_combis_with_q_metrics

