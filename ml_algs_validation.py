#DC - DisjunctiveCombination
#CC - ConjuctiveCombination
#SA - Single algs (length=1)
#MC - MAJORITY 
#BAGC - Bagging
#STACKC - Stacking
#combis - Algs Combinations

#группа классов ищет перспективные комбинации алгоритмов ML посредством кросс-валидации и генерации комбинаций из поданных на вход алгоритмов-одиночек,
#но можно и не проводить генерацию, а тестировать то, что есть, т.е. подать на вход алгоритмы-одиночки, составленные бэггинги, стекинги..., однако генерация позволяет
#перебрать все возможные тривиальные комбинации, бэггинги, стекинги.., в то время как без генерации мы можем лишь проверить умозрительно составленные комбинации,
#генерация же позволяет находить лучшие комбинации из всех возможных в принципе.
#Логированием результатов также занимается именно данный класс, поскольку он обладает специфическими данными, которых нет, например, в вызывающем коде и было
#бы странно в тот код передавать логику логирования результатов. 

#Общий алгоритм работы:
#1) генерируем фолды,
#2) тесты с кросс-валидацией,
#3) фильтруем,
#3) логгируем.
#Таким образом, какие бы комбинации не тестировались, они должны быть совместимы с основными этапами (должным образом унифицированы).
#Тривиальные комбинации связаны логической операцией, не более того. Усложнённые комбинации не просто связывают результаты работы каждого алгоритма,
#обучение каждого алгоритма каждой комбинации может проходить по-разному из-за разницы в обучающих семплах, в параметрах, доступных признаках..., а
#значит сначала запомнить предсказания алгоритмов-одиночек, а потом их связать не получится.

#Проводить тестирование single_algs необходимо независимо от того исследуются комбинации или нет, потому что данные о работе алгоритмов-одиночек используются
#при фильтрации любых типов комбинаций

#Дочерние классы валидаторов не сделаны статическими, поскольку должна быть возможность одновременной работы с разными экземплярами исследований.

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier

from enum import Enum
import time
import copy
from abc import ABC, abstractmethod, abstractstaticmethod
from multiprocessing import Process, Pool, Manager, JoinableQueue
from queue import Queue
import random
import threading
import math

from logs import *
from generic import *

class AlgsCombinationsValidator(): #алгоритмические комбинации должны быть dict-ами, ключами явл. названия, под это приспособлен код
    #1. Folds generation.
    #2. Make algs subsets.
    #3. Run different combis types validation.
    #4. Results post-processing.
    #5. Results logging.
    class AlgsCombination(ABC):
        def __init__(self, algs_names, algs_objs):
            self.quality_metrics = None #{}
            self.algs_names = algs_names #[]
            self.algs_objs = algs_objs #[]
            self.size = len(algs_names)

        @abstractmethod
        def create_name(self):
            ...

    _folds_splits = None
    _enabled_combis_types = None
    _det_metrics_exported_vals_decimal_places = 4
    _perf_metrics_exported_vals_decimal_places = 7
    _max_combis_lengths_dict = None

    _DETECTION_METRICS = ['f1', 'auc', 'acc', 'prec', 'rec'] #единые для всех комбинаций метрики, вынесены в отдельное место
    #для упрощения добавлений/удалений метрик из вычислений, унификации и наглядности
    _PERFOMANCE_METRICS = ['train_time', 'pred_time']


    @staticmethod
    def run(X, y, k_folds, algs_dicts, enabled_combinations_types, max_combis_lengths_dict, 
            det_metrics_exported_vals_decimal_places = 4, perf_metrics_exported_vals_decimal_places = 7): #совмещение исследований всех валидаторов
        AlgsCombinationsValidator._folds_splits = DatasetInstruments.make_stratified_split_on_stratified_k_folds(X,y,k_folds)
        AlgsCombinationsValidator._det_metrics_exported_vals_decimal_places = det_metrics_exported_vals_decimal_places
        AlgsCombinationsValidator._perf_metrics_exported_vals_decimal_places = perf_metrics_exported_vals_decimal_places
        AlgsCombinationsValidator._max_combis_lengths_dict = max_combis_lengths_dict
        AlgsCombinationsValidator._enabled_combis_types = enabled_combinations_types

        acv = AlgsCombinationsValidator()
        tcv = TrivialCombinationsValidator(algs_dicts)
        tcv._validate()
        ccv = ComplexCombinationsValidator(algs_dicts)
        ccv._validate()
        acv._postprocess_and_export_results(tcv, ccv)

    @abstractmethod
    def _validate(self, **params): #запуск валидатора, но без постпроцессинга и логирования
        ...

    def _postprocess_and_export_results(self, tcv, ccv = None): #этапы пост-обработки и экспорта объединены из-за их тесной связности
        def export_united_results_for_all_combis_types():
            self._export_results(all_types_combis_dict, 3, tcv._algs_SA, enabled_removers_DC_CC_MC, enabled_highlighters)

        def postprocess_and_export_DC_CC_MC_BAGC_BOOSTC(validator_obj, algs_SA_full, algs_SA_filtered, algs_combis, logger_num, enabled_removers, combis_type_cls = None):
            algs_combis_filtered = validator_obj._postprocess_results(algs_combis, algs_SA_full, enabled_removers, combis_type_cls)
            all_types_combis_dict.update(algs_combis_filtered)
            #все комбинации в своём логе должны быть смешаны и сравнены с обычными алгоритмами-одиночками - так будет наглядно показан смысл комбинирования или его
            #отсутствие
            validator_obj._export_results({**algs_SA_filtered,**algs_combis_filtered}, logger_num, algs_SA_full, enabled_removers, enabled_highlighters)

        #здесь должны быть указаны все возможные ремуверы и хайлайтеры, далее код сам разберётся что он может использовать, а что нет
        enabled_removers_DC_CC_MC = {'combis_with_bad_metrics_vals_remover': True,
        'useless_combis_remover': True,
        'excessively_long_combis_remover': True} #параметры вынесены из-за того, что фильтры и ремуверы отделены друг от друга и 
        #в логи тоже необходимо сообщить какие параметры использовались при ремувинге
        enabled_removers_SA = {'combis_with_bad_metrics_vals_remover': False,
        'useless_combis_remover': False,
        'excessively_long_combis_remover': False}
        enabled_removers_BAGC_BOOSTC = {'combis_with_bad_metrics_vals_remover': True,
        'useless_combis_remover': True,
        'excessively_long_combis_remover': False} #включенность и выключенность каких-то ремуверов зависит от необходимости и технической возможности

        enabled_highlighters = { 'best_combis_with_unique_algs_filter': True }
        enabled_highlighters_SA = { 'best_combis_with_unique_algs_filter': False }
        
        all_types_combis_dict = {}

        algs_SA_filtered = tcv._postprocess_results(tcv._algs_SA, tcv._algs_SA, enabled_removers_SA, TrivialCombisFiltration)
        all_types_combis_dict.update(algs_SA_filtered)
        tcv._export_results(algs_SA_filtered, 4, tcv._algs_SA, enabled_removers_SA, enabled_highlighters_SA)
        
        if AlgsCombinationsValidator._enabled_combis_types['DC']:
            postprocess_and_export_DC_CC_MC_BAGC_BOOSTC(tcv, tcv._algs_SA, algs_SA_filtered, tcv._algs_DC, 2, enabled_removers_DC_CC_MC, TrivialCombisFiltration)
        if AlgsCombinationsValidator._enabled_combis_types['CC']:
            postprocess_and_export_DC_CC_MC_BAGC_BOOSTC(tcv, tcv._algs_SA, algs_SA_filtered, tcv._algs_CC, 1, enabled_removers_DC_CC_MC, TrivialCombisFiltration)
        if AlgsCombinationsValidator._enabled_combis_types['MC']:
            postprocess_and_export_DC_CC_MC_BAGC_BOOSTC(tcv, tcv._algs_SA, algs_SA_filtered, tcv._algs_MC, 5, enabled_removers_DC_CC_MC, TrivialCombisFiltration)
        if AlgsCombinationsValidator._enabled_combis_types['BAGC']:
            postprocess_and_export_DC_CC_MC_BAGC_BOOSTC(ccv, tcv._algs_SA, algs_SA_filtered, ccv._algs_BAGC, 6, enabled_removers_BAGC_BOOSTC, BaggingCombisFiltration)
        export_united_results_for_all_combis_types()

    @abstractmethod
    def _log_with_highlighters(self, logger, sorted_algs_combis_di, algs_SA, enabled_highlighters):
        if enabled_highlighters['best_combis_with_unique_algs_filter']:
            filtered_results = CombinationsFiltration.highlight_best_unique_algs_combis_in_results(algs_SA, sorted_algs_combis_di)
            LogsFileProvider.log_named_info_block(logger, AlgsCombinationsValidator._make_searcher_results_str(filtered_results), 
                                                log_header="//// BEST UNIQUE COMBINATIONS ////")
    @abstractmethod
    def _init_combination(self, **args): #принято решение использовать args,поскольку нет фиксированного набора параметров
        ...
    
    @staticmethod
    def _remove_bad_combis(combis_dict_filtered, combis_dict_full, algs_SA, #управляет всеми возможными ремуверами
                enabled_removers, filters_type_cls):
        if enabled_removers['useless_combis_remover']:
            filters_type_cls.remove_useless_combis(combis_dict_filtered, algs_SA)
        if enabled_removers['combis_with_bad_metrics_vals_remover']:
            filters_type_cls.remove_combis_with_bad_metrics_vals(combis_dict_filtered)
        if enabled_removers['excessively_long_combis_remover']:
            filters_type_cls.remove_excessively_long_combinations(combis_dict_filtered, combis_dict_full, algs_SA)

    @staticmethod
    def _make_searcher_results_str(algs_combis_dict_items): #предполагается, что все типы комбинаций имеют одинаковый вид результатов
        str_ = ''
        for combi_name,combi_obj in algs_combis_dict_items:
            str_ += '---' + combi_name + '\n'
            str_ += str([metric_name + '=' + str(metric_val) for metric_name,metric_val in combi_obj.quality_metrics.items()]) + '\n'
        return str_

    @staticmethod
    def _sort_algs_combis_by_q_metrics(algs_combis_dict_items, criterias = [('f1', True)]):
        def multisort(dict_items):
            for key, enable_reverse in reversed(criterias):
                #dict_.sort(key=lambda el: el.quality_metrics[key], reverse=enable_reverse)
                dict_items = sorted(dict_items, key=lambda el: el[1].quality_metrics[key], reverse=enable_reverse)
            return dict_items
        return  multisort(algs_combis_dict_items)

    @staticmethod
    def _make_str_about_enabled_removers(enabled_removers):
        s = ''
        s += 'Enabled removers: \n'
        for remover_name in enabled_removers:
            s += "  " + remover_name + ': ' + str(enabled_removers[remover_name]) + '\n'
        s += "(These removers do not work for all types of combinations (despite True)) \n"
        return s

    @staticmethod
    def _log_filtered_results_info_block(filter_func, filter_params={}, log_header=LogsFileProvider.LOG_CONTENT_UNKN_HEADER):
        #экспортируем инфоблок с результатами с хедером в лог
        filtered_results = filter_func(**filter_params)
        LogsFileProvider.log_named_info_block(logger, AlgsCombinationsValidator._make_searcher_results_str(filtered_results), 
                                                filter_params, log_header)
    
    @staticmethod
    def _switch_loggers(results_from):
        lfp = LogsFileProvider()
        if (results_from == 1):
            return lfp.loggers['ml_CC_sorted_f1'], lfp.loggers['ml_CC_sorted_recall']
        if (results_from == 2):
            return lfp.loggers['ml_DC_sorted_f1'], lfp.loggers['ml_DC_sorted_recall']
        if (results_from == 3):
            return lfp.loggers['ml_ALL_sorted_f1'], lfp.loggers['ml_ALL_sorted_recall']
        if (results_from == 4):
            return lfp.loggers['ml_SA_sorted_f1'], None
        if (results_from == 5):
            return lfp.loggers['ml_MC_sorted_f1'], lfp.loggers['ml_MC_sorted_recall']
        if (results_from == 6):
            return lfp.loggers['ml_BAGC_sorted_f1'], lfp.loggers['ml_BAGC_sorted_recall']
    def _export_results(self, algs_combis_dict, results_from, algs_SA, enabled_removers, enabled_highlighters): 
        
        #enable_best_combis_with_unique_algs_filter = True, если сортировка только по убыванию качества
        #results - {algs_combi_name, {metrics}}
        
        #далее рассматривается dict как dict_items, т.к. для сортировки нужно задать новый порядок - его невозможно
        #задать в словаре, нужен другой его вид - список, не являющийся копией словаря. Даже если сортировка не будет производиться,
        #весь код далее для универсальности необходимо оставить адаптированным к dict_items
        #Ремуверам ни к чему работать в таком виде, потому что для них не имеет значения порядок комбинаций, 
        #фильтры - другое дело.
        algs_combis_dict_items = algs_combis_dict.items()
        f1_logger, recall_logger = self._switch_loggers(results_from) #если logger == null, то лог просто не выведется, ошибки не будет
        self._export_sorted_by(f1_logger, [('f1', True), ('rec', True), ('pred_time', False)], algs_combis_dict_items, algs_SA, enabled_removers, enabled_highlighters)
        self._export_sorted_by(recall_logger, [('rec', True), ('f1', True), ('pred_time', False)], algs_combis_dict_items, algs_SA, enabled_removers, enabled_highlighters)

    def _roundOff_metrics_of_combis(self, algs_combis_dict):
        for combi_name in algs_combis_dict:
            CollectionsInstruments.round_dict_vals(algs_combis_dict[combi_name].quality_metrics, 
                                                   AlgsCombinationsValidator._det_metrics_exported_vals_decimal_places, self._DETECTION_METRICS)
            CollectionsInstruments.round_dict_vals(algs_combis_dict[combi_name].quality_metrics, 
                                             AlgsCombinationsValidator._perf_metrics_exported_vals_decimal_places, self._PERFOMANCE_METRICS)
    
    def _postprocess_results(self, algs_combis, algs_SA_full, enabled_removers, combis_type_cls):
            #Для экспорта используются копии объектов, поскольку исследовательские данные и данные для экспорта, особенно при использовании
    #фильтров, парой сильно отличаются, но для той же фильтрации нужны именно исходные, т.е. исследовательские данные - их
    #нельзя менять или удалять. К тому же, должна быть возможность всегда использовать именно исследовательские данные,
    #а не адаптированные версии под экспорт, даже после самой процедуры экспорта, это делает код менее зависимым от очередности.
        algs_combis_filtered = copy.deepcopy(algs_combis)
        #фильтрация производится на основе округлённых значений, нет смысла разделять работу фильтрации и 
    #печати результатов
        self._roundOff_metrics_of_combis(algs_combis_filtered)
        AlgsCombinationsValidator._remove_bad_combis(algs_combis_filtered, algs_combis, algs_SA_full, enabled_removers, combis_type_cls) 
        return algs_combis_filtered
    
    def _export_sorted_by(self, logger, criterias_list, algs_combis_dict_items, algs_SA, enabled_removers, enabled_highlighters):
        if (logger == None):
            #print('logger == None')
            return

        logger.info("Sorted by: " + str(criterias_list))
        logger.info(AlgsCombinationsValidator._make_str_about_enabled_removers(enabled_removers))
        logger.info("") #enter

        sorted_algs_combis_di = AlgsCombinationsValidator._sort_algs_combis_by_q_metrics(algs_combis_dict_items, 
                                                                            criterias = criterias_list)
            
        self._log_with_highlighters(logger, sorted_algs_combis_di, algs_SA, enabled_highlighters)
        LogsFileProvider.log_named_info_block(logger, AlgsCombinationsValidator._make_searcher_results_str(sorted_algs_combis_di), 
                                    log_header="//// ALL COMBINATIONS ////")

    def _init_combis(self, algs_dict_items, combis_dict, combis_type_cls, combis_type = None, 
                          min_length = 1, max_length = 4): #add in-place
        algs_subsets = MathInstruments.make_subsets(algs_dict_items, max_length)#для остального кода важно, чтобы subset-ы
                #алгоритмов укладывались последовательно по мере увеличения длины комбинаций: сначала длин 1, потом 2...
                #[ [(alg_i_name, alg_i_obj),(),()...],[],[]... ]
        for subset in algs_subsets: 
            #в зависимости от min/max_length фильтруем список комбинаций
            if (len(subset) >= min_length and len(subset) <= max_length):
                algs_names, algs_objs = zip(*subset)
                algs_names, algs_objs = list(algs_names), list(algs_objs)
                algs_combi = self._init_combination(algs_names=algs_names, algs_objs=algs_objs, combi_type_cls=combis_type_cls, combi_type=combis_type)
                combis_dict[algs_combi.create_name()] = algs_combi

    @staticmethod
    def _calc_detection_quality_metrics_on_fold(y_pred, y_valid):
        acc = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)
        prec = precision_score(y_valid, y_pred)
        rec = recall_score(y_valid, y_pred)
        auc = roc_auc_score(y_valid, y_pred)
        return CollectionsInstruments.create_dict_by_keys_and_vals(AlgsCombinationsValidator._DETECTION_METRICS,[f1, auc, acc, prec, rec])

    @staticmethod
    def _calc_mean_det_metrics_vals_for_combi(combi_det_q_metrics_on_folds_dicts):
        #Раскомментировать для логирования
        #LogsFileProvider().loggers['ml_research_calculations'].info('--------- Проверка правильности подсчёта метрик ---------')
        n = len(combi_det_q_metrics_on_folds_dicts)
        combi_mean_det_q_metrics_dict = CollectionsInstruments.create_dict_by_keys_and_vals(AlgsCombinationsValidator._DETECTION_METRICS,
                                                                                           [0 for _ in AlgsCombinationsValidator._DETECTION_METRICS])
        for metrics_dict in combi_det_q_metrics_on_folds_dicts:
            #Раскомментировать для логирования
            #LogsFileProvider().loggers['ml_research_calculations'].info(metrics)
            metrics_on_fold = metrics_dict.values()
            combi_mean_det_q_metrics_dict = { metric_name:metric_val+new_metric_val/n for (metric_name, metric_val),new_metric_val in 
                                            zip(combi_mean_det_q_metrics_dict.items(), metrics_on_fold) } 
            #среднее значение каждой метрики
            
        #Раскомментировать для логирования
        #LogsFileProvider().loggers['ml_research_calculations'].info('--- Итоговая метрика' + str(mean_det_q_metrics_for_combi))
        return combi_mean_det_q_metrics_dict

class ComplexCombinationsValidator(AlgsCombinationsValidator):
    #отличие от валидатора тривиальных, данный не предусматривает запоминание результатов предсказаний алгоритмов-одиночек и их последующее комбинирование,
    #т.к. алгоритмы-участники в сложных комбинациях от комбинации к комбинации находятся в разных условиях, а не фиксированных 
    class ComplexCombination(AlgsCombinationsValidator.AlgsCombination, ABC):
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)
            self._clf = None

        def fit(self, X, y):
            self._clf.fit(X, y)

        def predict(self, X):
            return self._clf.predict(X)
        
        @abstractmethod
        def create_name(self, combi_type_name):
            return combi_type_name + ': ' + str(self.algs_names)

    class BaggingCombination(ComplexCombination): #n_jobs = -1 не работает
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)
            self._clf = BaggingClassifier(base_estimator=algs_objs[0].clf, n_estimators=AlgsCombinationsValidator._max_combis_lengths_dict['BAGC'], 
                                          n_jobs = 1, bootstrap = True, max_samples=0.95)

        def create_name(self):
            return super().create_name('BAGGING')

    class BoostingCombination(ComplexCombination):
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)
            

        def create_name(self):
            return super().create_name('BOOSTING')

    class StackingCombination(ComplexCombination): 
        #algs_names=[[{alg_names_layer1}],[],..,[]]
        #algs_objs=[[{algs_layer1}],[],..,[]]
        #size - tuple(algs_layer1_amount,..)
        #генерящийся стекинг представляет собой слои, в каждом есть алгоритмы, логику взаимодействия слоёв задают fit() и predict(),
        #возможно ещё какие-то вспомогательные методы, которые должны быть определены дополнительно
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)

        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

        def create_name(self):
            return super().create_name('STACKING')

    def _init_combination(self, **args):
        return args['combi_type_cls'](args['algs_names'], args['algs_objs'])

    def __init__(self, algs_dicts):
        if AlgsCombinationsValidator._enabled_combis_types['BAGC']: #для бэггинга нужен другой метод инициализаци (init_BAGC_BOOSTC()) и, поскольку не сабсеты используются, а дублирующиеся алгоритмы одиночки
            #такой метод подойдёт и для бустинга,поэтому его стоит определить только в рамках complex validator-а. 
            self._algs_BAGC = {}
            self._init_combis(algs_dicts['BAGC'].items(), self._algs_BAGC, self.BaggingCombination, min_length = 1, max_length=1)
        if AlgsCombinationsValidator._enabled_combis_types['BOOSTC']:
            self._algs_BOOSTC = {}
            self._init_combis(algs_dicts['BOOSTC'].items(), self._algs_BOOSTC, self.BoostingCombination, min_length = 1, max_length=1)
        #if AlgsCombinationsValidator._enabled_combis_types['STACKC']:
        #    self._algs_STACKC = {}
        #    self._init_combis(algs_dicts['STACKC'].items(), self._algs_STACKC, self.StackingCombination, min_length = 2)

    def _validate(self, **params):
        if AlgsCombinationsValidator._enabled_combis_types['BAGC']:
            self.__validate_combis_on_folds(self._algs_BAGC)
            print('////////////////// BAGC validation done')
        #if AlgsCombinationsValidator._enabled_combis_types['BOOSTC']:
            
        #if AlgsCombinationsValidator._enabled_combis_types['STACKC']:
    
    def __validate_combis_on_folds(self, combis_dict):
        for combi_name in combis_dict:
            train_times_on_folds = []
            pred_time_on_folds = []
            combi_det_q_metrics_on_folds_dicts = []

            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in AlgsCombinationsValidator._folds_splits:
                t0 = time.time()
                combis_dict[combi_name].fit(X_trainFolds, y_trainFolds)
                t1 = time.time()
                y_pred_on_validFold = combis_dict[combi_name].predict(X_validFold)
                t2 = time.time()
                train_times_on_folds.append(t1-t0)
                pred_time_on_folds.append(t2-t1)
                combi_det_q_metrics_on_folds_dicts.append(AlgsCombinationsValidator._calc_detection_quality_metrics_on_fold(y_pred_on_validFold, y_validFold))
            combi_mean_det_q_metrics = AlgsCombinationsValidator._calc_mean_det_metrics_vals_for_combi(combi_det_q_metrics_on_folds_dicts)
            combi_mean_perf_q_metrics = CollectionsInstruments.create_dict_by_keys_and_vals(AlgsCombinationsValidator._PERFOMANCE_METRICS, 
                                                                                      [np.mean(train_times_on_folds), np.mean(pred_time_on_folds)])
            combis_dict[combi_name].quality_metrics = CollectionsInstruments.merge_dicts(combi_mean_det_q_metrics, combi_mean_perf_q_metrics)

    def _log_with_highlighters(self, logger, sorted_algs_combis_di, algs_SA, enabled_highlighters):
        super()._log_with_highlighters(logger, sorted_algs_combis_di, algs_SA, enabled_highlighters)

class TrivialCombinationsValidator(AlgsCombinationsValidator):
    #Такие комбинации представляют собой алгоритмы, результаты работы которых объединены агрегирующим законом (все за одного, большинства...),
        #больше нет никаки особенностей связи и процесса обучения. Значит нам достаточно сохранить результаты работы каждого алгоритма-одиночки на фолдах,
        #а потом агрегировать их по-разному в зависимости от закона.
    class TrivialCombination(AlgsCombinationsValidator.AlgsCombination):
        class Types(Enum):
            SINGLE = 0
            DISJUNCTIVE = 1
            CONJUNCTIVE = 2
            MAJORITY = 3

        class TrivialCombiTask(object): #комбинация, поданная на многопоточную обработку
            def __init__(self, combi_name, combi_obj, SA_y_preds, folds_splits, SA_algs, multiproc_validator):
                self.SA_y_preds = SA_y_preds
                self.SA_algs = SA_algs
                self.folds_splits = folds_splits
                self.combi_obj = combi_obj
                self.combi_name = combi_name
                self.multiproc_validator_obj = multiproc_validator
            

        #методы predict и fit не были имплементированы, поскольку результаты работы комбинации рассчитываются
        #исходя из комбинирования результатов работы алгоритмов-одиночек
        def __init__(self, type, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)
            self.type = type

        @staticmethod
        def create_combi_name(combi_type, combi_algs_names):
            SEPARATORS_FOR_COMBI_NAMES = { TrivialCombinationsValidator.TrivialCombination.Types.SINGLE: '  ',
                                TrivialCombinationsValidator.TrivialCombination.Types.DISJUNCTIVE: ' + ', 
                                    TrivialCombinationsValidator.TrivialCombination.Types.CONJUNCTIVE: ' * ',
                                   TrivialCombinationsValidator.TrivialCombination.Types.MAJORITY: ' <-> '}
            combi_name = ''
            separator_for_name = SEPARATORS_FOR_COMBI_NAMES[combi_type]
            for alg_name in combi_algs_names:
                combi_name += alg_name + separator_for_name
            return combi_name[0:-len(separator_for_name)]

        def create_name(self):
            return self.create_combi_name(self.type, self.algs_names)

    def __init__(self, algs_dicts):
        self._algs_SA = {} #алгоритмы по одиночке (длина комбинации = 1)
        self._init_combis(algs_dicts['SA'].items(), self._algs_SA, self.TrivialCombination, self.TrivialCombination.Types.SINGLE, min_length=1, 
                                   max_length=1)

        if AlgsCombinationsValidator._enabled_combis_types['DC']:
            self._algs_DC = {}
            self._init_combis(algs_dicts['DC'].items(), self._algs_DC, self.TrivialCombination, self.TrivialCombination.Types.DISJUNCTIVE, min_length=2, 
                                   max_length=AlgsCombinationsValidator._max_combis_lengths_dict['DC'])
        if AlgsCombinationsValidator._enabled_combis_types['CC']:
            self._algs_CC = {}
            self._init_combis(algs_dicts['CC'].items(), self._algs_CC, self.TrivialCombination, self.TrivialCombination.Types.CONJUNCTIVE, min_length=2, 
                                   max_length=AlgsCombinationsValidator._max_combis_lengths_dict['CC'])
        if AlgsCombinationsValidator._enabled_combis_types['MC']:
            self._algs_MC = {}
            self._init_combis(algs_dicts['MC'].items(), self._algs_MC, self.TrivialCombination, self.TrivialCombination.Types.MAJORITY, min_length=2, 
                                   max_length=AlgsCombinationsValidator._max_combis_lengths_dict['MC'])

    def _validate(self, **args):
        single_algs_y_preds_on_folds = self.__validate_single_algs_on_folds()
        if AlgsCombinationsValidator._enabled_combis_types['DC']:
            self.__run_combis_validation(single_algs_y_preds_on_folds, self.TrivialCombination.Types.DISJUNCTIVE, multi_threading = False)
            print('////////////////// DC validation done')
        if AlgsCombinationsValidator._enabled_combis_types['CC']:
            self.__run_combis_validation(single_algs_y_preds_on_folds, self.TrivialCombination.Types.CONJUNCTIVE, multi_threading = False)
            print('////////////////// CC validation done')
        if AlgsCombinationsValidator._enabled_combis_types['MC']:
            self.__run_combis_validation(single_algs_y_preds_on_folds, self.TrivialCombination.Types.MAJORITY, multi_threading = False)
            print('////////////////// MC validation done')

    def __validate_single_algs_on_folds(self): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
        single_algs_y_pred = {} #метод не стоит делать многопоточным из-за присутствия в тестах итак распараллеленых алгоритмов

        for alg_name in self._algs_SA:
            alg_obj = self._algs_SA[alg_name]
            train_times_on_folds = []
            pred_time_on_folds = []
            single_algs_y_pred[alg_obj.create_name()] = []
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in AlgsCombinationsValidator._folds_splits:
                #time.time() - показывает реальное время, а не время CPU, поэтому результаты не очень точные, но этой точности
                #достаточно, для реализации более точного подсчета с timeit нужно писать куда больше кода. Также, нам не важно
                #получить абсолютные достоверные значения, важно дифференцировать алгоритмы друг с другом.
                t0 = time.time()
                alg_obj.algs_objs[0].learn(X_train = X_trainFolds, y_train = y_trainFolds)
                t1 = time.time()
                y_pred_alg = alg_obj.algs_objs[0].predict(X_test = X_validFold)
                t2 = time.time()
                single_algs_y_pred[alg_name].append(y_pred_alg)
                train_times_on_folds.append(t1-t0)
                pred_time_on_folds.append(t2-t1)
            perf_q_metrics = CollectionsInstruments.create_dict_by_keys_and_vals(self._PERFOMANCE_METRICS,
                                                                                [np.mean(train_times_on_folds), np.mean(pred_time_on_folds)])
            det_q_metrics = self._calc_mean_det_q_metrics_on_folds_for_algs_combi(alg_obj, single_algs_y_pred[alg_name], AlgsCombinationsValidator._folds_splits)
            alg_obj.quality_metrics = CollectionsInstruments.merge_dicts(det_q_metrics, perf_q_metrics)
        print('////////////////// test_single_algs_on_folds() done')
        return single_algs_y_pred

    @staticmethod
    def _calc_mean_det_q_metrics_on_folds_for_algs_combi(combi_obj, y_pred_combi_on_folds, folds_splits):
        combi_det_q_metrics_on_folds_dicts = []
        for (i,(_, _, _, y_validFold)) in enumerate(folds_splits):
            combi_det_q_metrics_on_folds_dicts.append(AlgsCombinationsValidator._calc_detection_quality_metrics_on_fold(y_pred_combi_on_folds[i], y_validFold))
        return AlgsCombinationsValidator._calc_mean_det_metrics_vals_for_combi(combi_det_q_metrics_on_folds_dicts)

    @staticmethod
    def _calc_perf_q_metrics_for_algs_combi(combi_obj, algs_SA):
        #для быстрого обращения к нужным данным
        single_algs_perf_metrics = {alg_name:{perf_metric_name: algs_SA[alg_name].quality_metrics[perf_metric_name] 
                                                        for perf_metric_name in AlgsCombinationsValidator._PERFOMANCE_METRICS} for alg_name in algs_SA}
        #{algs_combi_name:{'train_time':, 'pred_time': }}
        #pred_time, train_time вычисляются для комбинации подсчетом суммы значений этих метрик для каждого алгоритма в отдельности,
        #не учитывается время, которое тратится на агрегацию прогнозов (or, and функции, например), но это и не важно
        dicts = [single_algs_perf_metrics[alg_name] for alg_name in combi_obj.algs_names]
        #print(dicts)
        return CollectionsInstruments.sum_vals_of_similar_dicts(dicts, AlgsCombinationsValidator._perf_metrics_exported_vals_decimal_places)

    def _get_combis_by_type(self, combis_type):
        if combis_type == self.TrivialCombination.Types.CONJUNCTIVE: 
            return self._algs_CC
        if combis_type == self.TrivialCombination.Types.DISJUNCTIVE:
            return self._algs_DC
        if combis_type == self.TrivialCombination.Types.MAJORITY:
            return self._algs_MC
    
    @staticmethod
    def _calc_y_preds_combi_on_folds(combi_obj, SA_y_preds, folds_splits):
        def tune_validation(combis_type):
            def DC_agregation_func(y_pred_combination, y_pred_alg):
                return np.logical_or(y_pred_combination, y_pred_alg)
            def CC_agregation_func(y_pred_combination, y_pred_alg):
                return np.logical_and(y_pred_combination, y_pred_alg)
            def MC_agregation_func(y_pred_combination, y_pred_alg):
                #На k-1 фолдах копим данные о кол-ве вердиктом "спам" на каждый семпл.
                #на k-ом запоминаем последний вердикт и рассчитываем y_combi по принципу голосования за вердикт
                for sample_num,y_pred_i in enumerate(y_pred_alg):
                    combi_algs_spam_verdicts_on_fold[sample_num] += y_pred_i
                n_algs_in_combi = len(combi_obj.algs_names)
                if alg_num == n_algs_in_combi -1:
                    for sample_num in range(len(combi_algs_spam_verdicts_on_fold)):
                        if combi_algs_spam_verdicts_on_fold[sample_num]/n_algs_in_combi > 0.5: #голосов за "спам" больше
                            y_pred_combination[sample_num] = 1
                        if combi_algs_spam_verdicts_on_fold[sample_num]/n_algs_in_combi < 0.5:
                            y_pred_combination[sample_num] = 0
                        if combi_algs_spam_verdicts_on_fold[sample_num]/n_algs_in_combi == 0.5:
                            y_pred_combination[sample_num] = random.randint(0,1+1)
                return y_pred_combination

            combis_aggregation_func = None
            aggreg_func_args = {}
            y_pred_combi_init_func = None

            if combis_type == TrivialCombinationsValidator.TrivialCombination.Types.CONJUNCTIVE:                
                combis_aggregation_func = CC_agregation_func
                y_pred_combi_init_func = np.ones
            if combis_type == TrivialCombinationsValidator.TrivialCombination.Types.DISJUNCTIVE:
                combis_aggregation_func = DC_agregation_func
                y_pred_combi_init_func = np.zeros
            if combis_type == TrivialCombinationsValidator.TrivialCombination.Types.MAJORITY:
                combis_aggregation_func = MC_agregation_func
                y_pred_combi_init_func = np.zeros
            return combis_aggregation_func, y_pred_combi_init_func

        calc_y_pred_combi, y_pred_combi_init_func = tune_validation(combi_obj.type)
            
        #Раскомментировать для логирования
        #LogsFileProvider().loggers['ml_research_calculations'].info('---------' + str(self.get_algs_combination_name(combi)))
        y_pred_combi_on_folds = []
        for (folds_split_num,(_, _, _, y_validFold)) in enumerate(folds_splits):
            y_pred_combination = y_pred_combi_init_func(y_validFold.shape, dtype=bool)
            combi_algs_spam_verdicts_on_fold = [0 for _ in range(y_validFold.shape[0])] #каждому семплу соответсвует кол-во зафиксированных вердиктов "спам" (для MC)
            for alg_num, alg_name in enumerate(combi_obj.algs_names):
                #Раскомментировать для логирования
                #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                #LogsFileProvider().loggers['ml_research_calculations'].info('y_pred_combination before' + str(dict(zip(classes.tolist(), classes_counts))))
                y_pred_alg = SA_y_preds[alg_name][folds_split_num]
                y_pred_combination = calc_y_pred_combi(y_pred_combination, y_pred_alg)
                #Раскомментировать для логирования
                #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                #LogsFileProvider().loggers['ml_research_calculations'].info('y_pred_combination after' + str(dict(zip(classes.tolist(), classes_counts))))
            y_pred_combi_on_folds.append(y_pred_combination)

        return y_pred_combi_on_folds #[[],[],[]...]
    
    @staticmethod
    def validate_combi(combi_obj, SA_y_preds, folds_splits, algs_SA): #адаптирован под многопоточность
        y_pred_combi_on_folds = TrivialCombinationsValidator._calc_y_preds_combi_on_folds(combi_obj, SA_y_preds, folds_splits) #предсказания комбинации на фолдах
        combi_detection_q_metrics = TrivialCombinationsValidator._calc_mean_det_q_metrics_on_folds_for_algs_combi(combi_obj, y_pred_combi_on_folds, folds_splits)
        combi_perfomance_q_metrics = TrivialCombinationsValidator._calc_perf_q_metrics_for_algs_combi(combi_obj, algs_SA)
        combi_obj.quality_metrics = CollectionsInstruments.merge_dicts(combi_detection_q_metrics, combi_perfomance_q_metrics)

    def __run_combis_validation(self, SA_y_preds, combis_type, multi_threading = False):
        #multi_threading = True не работает, хотя функционал внедрён, есть неизвестные преграды, подробности в реализации
        def in_single_thread():
            for combi_name in algs_combis_dict:
                self.validate_combi(algs_combis_dict[combi_name], SA_y_preds, AlgsCombinationsValidator._folds_splits, self._algs_SA)

        algs_combis_dict = self._get_combis_by_type(combis_type)
        if multi_threading: 
            TrivialCombisMultiprocessingValidator.run(algs_combis_dict, SA_y_preds, AlgsCombinationsValidator._folds_splits, self._algs_SA)
        else:
            in_single_thread()

    def _log_with_highlighters(self, logger, sorted_algs_combis_di, algs_SA, enabled_highlighters):
        super()._log_with_highlighters(logger, sorted_algs_combis_di, algs_SA, enabled_highlighters)

    def _init_combination(self, **args):
        return self.TrivialCombination(args['combi_type'], args['algs_names'], args['algs_objs'])

class CombinationsFiltration(ABC): 
    #ремуверы (removers) - удаляют бесполезное и хайлайтеры (highlighter) - выделяют подмножество комбинаций опред. свойств
    # combis_dict_filtered - словарь с комбинациями, который проходит фильтрацию и не содержит первоначальный набор добытых комбинаций

    #многие ремуверы работают по единой схеме: идёт перебор комбинаций и удаляются те, которые не отвечают требованиям
    #наличие single_combis_compatibility не является избыточной мерой, каждый ремувер характеризуется совместимостью и не важно 
    #будет или нет внешний код сам понимать когда нужно выключать какие-либо фильтры во избежание проблем с совместимостью
    @staticmethod
    def _remove_combis_by_condition(condition_func, combis_dict_filtered, algs_SA_compatibility, params):
        #Раскомментировать для логирования
        #LogsFileProvider().loggers['ml_research_calculations'].info("///// "+str(condition_func))

        keys_removal_list=[]
        for combi_name in combis_dict_filtered:
            if type(combis_dict_filtered[combi_name]) is TrivialCombinationsValidator.TrivialCombination and not algs_SA_compatibility:
                if combis_dict_filtered[combi_name].type == TrivialCombinationsValidator.TrivialCombination.Types.SINGLE:
                    continue
            if condition_func(combis_dict_filtered[combi_name], **params):
                #Раскомментировать для логирования
                #LogsFileProvider().loggers['ml_research_calculations'].info(combi_name)
                keys_removal_list.append(combi_name)
        CollectionsInstruments.delete_dict_elements_by_removal_list(combis_dict_filtered, keys_removal_list)

    #Бесполезные комбинации - это такие, которые не превосходят в одной из 
    #основных метрик (recall, precision, pred_time*) все алгоритмы-участники по одиночке
    #*по факту не проверяется, поскольку комбинация всегда работает медленнее её участников по-одиночке,
    #поэтому если комбинация не лучше в prec или rec, то она считается бесполезной
    #если требуется уточнение "на сколько лучше", то менять необходимо кол-во знаков после запятой
    @staticmethod
    def _remove_useless_combis(combis_dict_filtered, algs_SA, comparing_metrics=['rec','prec']):
        def is_combi_useless(combi, comparing_metrics):
            def is_combi_metrics_not_better(algs_combi_metrics, single_algs_metrics): 
                #векторизированное сравнение, кол-во метрик может быть любым
                metrics_of_combi = np.array([algs_combi_metrics[metric] for metric in comparing_metrics])
                metrics_of_single_algs = [np.array([alg_metrics[metric] for metric in comparing_metrics])
                                            for alg_metrics in single_algs_metrics] #[[metrics_of_alg_i],[],[]]
                for single_alg_metrics in metrics_of_single_algs:
                    if not np.any(np.greater(metrics_of_combi, single_alg_metrics)):
                        return True
                return False
            single_algs_metrics = [copy.deepcopy(algs_SA[alg_name].quality_metrics)            
                                    for alg_name in combi.algs_names] #алгоритмы участники комбинации
            #в исходном виде (не для экспорта) в комбинациях хранятся метрики не округленные
            for alg_metrics in single_algs_metrics:
                CollectionsInstruments.round_dict_vals(alg_metrics, 
                                                        AlgsCombinationsValidator._det_metrics_exported_vals_decimal_places, AlgsCombinationsValidator._DETECTION_METRICS)
                CollectionsInstruments.round_dict_vals(alg_metrics, 
                                                        AlgsCombinationsValidator._perf_metrics_exported_vals_decimal_places, AlgsCombinationsValidator._PERFOMANCE_METRICS)
            return is_combi_metrics_not_better(combi.quality_metrics, single_algs_metrics)

        CombinationsFiltration._remove_combis_by_condition(is_combi_useless, combis_dict_filtered, False, {'comparing_metrics':comparing_metrics})

    #комбинации избыточной длинны - добавление нового алгоритма не улучшает никакие основные метрики
       #если к комбинации прикрепить ещё один и более алгоритмов, то комбинация останется избыточной,
            #т.е. удалять такие комбинации надо в идеале комплексно - с удалением подобных.
    @staticmethod
    def _remove_excessively_long_combinations(combis_dict_filtered, combis_dict_full, algs_SA,comparing_metrics=['rec','prec']):
        #2)T=Cn_k*n: перебираем каждую комбинацию, берём её список алгоритмов(m), на основе m-1 из них делаем сабсеты,
        #находим таких комбинаций-предшественников. Минус подхода: много дублирующих проверок и медлительность.
        #3)T=O(n*l)-комбинации длины 2 избыточные, l-длина комбинации, W(n) - все комбинации нормальные: 
        #перебираем каждую комбинацию (не по словарю, 
        #а делаем копию его ключей, чтобы удалять можно было в цикле), причем начинаем от коротких к длинным,
        #за итерацию мы определяем избыточна ли комбинация текущая, если да, то запускаем поиск в глубь избыточных на её основе,
        #попутно найденное добавляем в список на удаление, в самом конце итерации по списку удаляем ключи, продолжаем итерации (
        #естественно каких-то комбинаций не будет, которые мы ищем, пропускай такие случаи exception-ом) и т.д. так мы удалим все комбинации.
        #Здесь реализован алгоритм №3.
                
        def delete_child_combis(parent_combi_name):
            parent_combi_obj = combis_dict_filtered[parent_combi_name]
            #определяем доступные алгоритмы для генерации имён удаляемых комбинаций
            available_algs_names = all_algs_names - set(parent_combi_obj.algs_names)
            #составляем имена
            combis_names_generated = [TrivialCombinationsValidator.TrivialCombination.create_combi_name(parent_combi_obj.type, parent_combi_obj.algs_names + [name]) 
                                        for name in available_algs_names]
            #проверяем наличие таких комбинаций. если найдены, добавляем в список на удаление
            #делаем имена новыми родителями и вызываем данную функцию снова
            for combi_name in combis_names_generated:
                if combi_name in combis_dict_filtered:
                    keys_removal_list.append(combi_name)
                    delete_child_combis(combi_name)
        #Раскомментировать для логирования
        #LogsFileProvider().loggers['ml_research_calculations'].info("/////--------------- delete_excessively_long_combinations()")

        #зафиксировать все доступные алгоритмы (их имена) 
        all_algs_names = set([alg_name for alg_name in algs_SA])
        for combi_name in combis_dict_full:
            keys_removal_list = []
            if not combi_name in combis_dict_filtered:
                continue
            combi_obj = combis_dict_filtered[combi_name]
            if combi_obj.type == TrivialCombinationsValidator.TrivialCombination.Types.SINGLE:
                continue
            combi_predecessor_name = TrivialCombinationsValidator.TrivialCombination.create_combi_name(combi_obj.type, combi_obj.algs_names[:-1])
            if not combi_predecessor_name in combis_dict_filtered:
                continue
            combi_predecessor = combis_dict_filtered[combi_predecessor_name] if len(combi_obj.algs_names[:-1]) > 1 else algs_SA[combi_predecessor_name]
            combi_q_metrics_vals = [combi_obj.quality_metrics[metric] for metric in comparing_metrics]
            combi_pred_q_metrics_vals = [combi_predecessor.quality_metrics[metric] for metric in comparing_metrics]
            is_combi_excessively_long = not np.any(np.greater(np.array(combi_q_metrics_vals), 
                                                                np.array(combi_pred_q_metrics_vals)))
            if is_combi_excessively_long:
                keys_removal_list.append(combi_name)
                delete_child_combis(combi_name) #дочерние комбинации избыточны, если родитель избыточен
                CollectionsInstruments.delete_dict_elements_by_removal_list(combis_dict_filtered, keys_removal_list)
                #Раскомментировать для логирования
                #LogsFileProvider().loggers['ml_research_calculations'].info(str(keys_removal_list))

    @staticmethod
    def _remove_combis_with_bad_metrics_vals(combis_dict_filtered, not_bad_metr_vals_range={'prec':[0.9,1.0], 'rec':[0.85,1.0]}):
        def is_combi_has_bad_metrics_vals(combi, not_bad_metr_vals_range):
        #фильтрация по диапазонам метрик качества обнаружения
        #можно фильтровать по разным метрикам, не только указанным
            def is_combi_has_bad_metric_val(metric_name, vals_range):
                return not (vals_range[0] <= combi.quality_metrics[metric_name] <= vals_range[1])
                
            for metric_name, vals_range in not_bad_metr_vals_range.items(): #(metric_name, [])
                if is_combi_has_bad_metric_val(metric_name, vals_range):
                    return True
            return False
        CombinationsFiltration._remove_combis_by_condition(is_combi_has_bad_metrics_vals, combis_dict_filtered, True, {'not_bad_metr_vals_range': not_bad_metr_vals_range})

    @staticmethod
    def highlight_best_unique_algs_combis_in_results(algs_SA, sorted_combis_dict_items, entries_amount_of_each_alg = 1): 
    #entries_amount_of_alg = 1 is combis_with_unique_algs
    #Уникальная комбинация - та, где есть хотя бы один уникальный алгоритм (ранее не встречался в комбинациях).
    #Перебираем комбинации сверху вниз по логу результатов,
    #алгоритмы содержатся в dict-е, значением является то самое кол-во комбинаций, где алгоритм был обнаружен,
    #это значение не может превышать entries_amount_of_alg
    #фиксируем алгоритм, запоминаем индекс в sorted_results.
    #по индексам формируем список фильтрованных результатов, по факту они находятся в полном списке - лишняя память не расходуется
    #Данный фильтр хорош тогда, когда комбинации отсортированы по убыванию своего качества, на выходе мы получим
    #уникальные комбинации с лучшим качеством (по той метрике, по которой отсортирован лог).
    #Т.е. мы видим лучшие комбинации при уникальном составе алгоритмов, можем оценить влияние на более качественные комбинации (выше)
    #добавленых др. алгоритмов.
        def mark_algs_entries_in_combi():
            for alg_name in combi_obj.algs_names:
                algs_entries_in_combis[alg_name] += 1

        def is_combi_content_unique_alg():
            for alg_name in combi_obj.algs_names:
                if algs_entries_in_combis[alg_name] == entries_amount_of_each_alg:
                    return True
            return False

        combis_with_unique_algs = []
        algs_entries_in_combis = dict(zip([single_alg_name for single_alg_name in algs_SA], [0 for _ in algs_SA]))
        for combi_name, combi_obj in sorted_combis_dict_items:
            mark_algs_entries_in_combi()
            if is_combi_content_unique_alg():
                combis_with_unique_algs.append((combi_name,combi_obj))
        return combis_with_unique_algs

class TrivialCombisFiltration(CombinationsFiltration, ABC):
    @staticmethod
    def highlight_best_unique_algs_combis_in_results(algs_SA, sorted_combis_dict_items, entries_amount_of_each_alg = 1):
        CombinationsFiltration.highlight_best_unique_algs_combis_in_results(algs_SA, sorted_combis_dict_items, entries_amount_of_each_alg)

    @staticmethod
    def remove_excessively_long_combinations(algs_combis_filtered, algs_combis, algs_SA, comparing_metrics=['rec','prec']):
        CombinationsFiltration._remove_excessively_long_combinations(algs_combis_filtered, algs_combis, algs_SA, comparing_metrics)
            
    @staticmethod
    def remove_combis_with_bad_metrics_vals(combis_dict_filtered):
        CombinationsFiltration._remove_combis_with_bad_metrics_vals(combis_dict_filtered)

    @staticmethod
    def remove_useless_combis(combis_dict_filtered, algs_SA):
        CombinationsFiltration._remove_useless_combis(combis_dict_filtered, algs_SA)

class BaggingCombisFiltration(CombinationsFiltration, ABC):
    @staticmethod
    def highlight_best_unique_algs_combis_in_results(algs_SA, sorted_combis_dict_items, entries_amount_of_each_alg = 1):
        CombinationsFiltration.highlight_best_unique_algs_combis_in_results(algs_SA, sorted_combis_dict_items, entries_amount_of_each_alg)

    @staticmethod
    def remove_combis_with_bad_metrics_vals(combis_dict_filtered):
        CombinationsFiltration._remove_combis_with_bad_metrics_vals(combis_dict_filtered)

    @staticmethod
    def remove_useless_combis(combis_dict_filtered, algs_SA):
        CombinationsFiltration._remove_useless_combis(combis_dict_filtered, algs_SA)


class StackingCombisFiltration(CombinationsFiltration, ABC):
    pass

class BoostingCombisFiltration(CombinationsFiltration, ABC):
    pass


#Многопоточность не работает, исключение смотреть при компиляции без дебага (Ctrl+F5)
def validate_combis(tasks_queue, lock): #target-функция должна быть на верхнем уровне модуля
    for task in tasks_queue:
        TrivialCombinationsValidator.validate_combi(task.combi_obj,task.SA_y_preds, task.folds_splits, task.SA_algs)
        with lock:
            TrivialCombisMultiprocessingValidator.add_combi(task.combi_name,task.combi_obj, task.multiproc_validator_obj)  

class TrivialCombisMultiprocessingValidator(object):
    #Каждый поток получает заранее свою очередь задач
    @staticmethod
    def run(algs_combis_dict, SA_y_preds, folds_splits, SA_algs):
        self_ = TrivialCombisMultiprocessingValidator()
        self_.algs_combis_dict = algs_combis_dict
        self_.SA_y_preds = SA_y_preds
        self_.folds_splits = folds_splits
        self_.SA_algs = SA_algs

        algs_combis_copy = copy.deepcopy(self_.algs_combis_dict)
        CollectionsInstruments.delete_dict_elements_by_removal_list(self_.algs_combis_dict, list(self_.algs_combis_dict.keys()))
        lock = multiprocessing.Lock()
        n_threads = ServiceInstruments.calc_optimal_threads_amount()
        self_.tasks_subsets = self_._split_tasks_on_workers(algs_combis_copy, n_threads)
        print('used threads:', n_threads)
        self_.workers = [multiprocessing.Process(target=validate_combis, args=(tasks_split, lock, )) for tasks_split in self_.tasks_subsets]
        self_._run_workers()
        self_._workers_join()
    
    def _split_tasks_on_workers(self, algs_combis_dict, n_workers):
        tasks_list = self._init_tasks(algs_combis_dict)
        n_tasks = len(tasks_list)
        n_tasks_on_thread = math.ceil(n_tasks / n_workers)
        tasks_queues = []
        for i in range(min(n_workers, n_tasks)):
            task_num_lower = i * n_tasks_on_thread
            task_num_upper = min((i+1) * n_tasks_on_thread, n_tasks)
            if task_num_lower + 1 == task_num_upper:
                break
            tasks_queue = []
            for task_num in range(task_num_lower,task_num_upper):
                tasks_queue.append(tasks_list[task_num])
            tasks_queues.append(tasks_queue)
        return tasks_queues

    def _run_workers(self):
        for w in self.workers:
            w.start()

    def _workers_join(self):
        for w in self.workers:
            w.join()
        

    @staticmethod
    def add_combi(combi_name, combi_obj, self_): #single threading
        self_.algs_combis_dict[combi_name] = combi_obj

    def _init_tasks(self, algs_combis):
        tasks = []
        for combi_name in algs_combis:
            task = TrivialCombinationsValidator.TrivialCombination.TrivialCombiTask(
                combi_name, algs_combis[combi_name],copy.deepcopy(self.SA_y_preds), copy.deepcopy(self.folds_splits), copy.deepcopy(self.SA_algs), self)
            tasks.append(task)
        return tasks

#Версия с проблемой, которую не удалось решить
#class TrivialCombisMultiprocessingValidator(object):
#    #реализация шаблона Producer-Consumers
#    @staticmethod
#    def run(algs_combis_dict, SA_y_preds, folds_splits, SA_algs):
#        self_ = TrivialCombisMultiprocessingValidator()
#        self_.algs_combis_dict = algs_combis_dict
#        self_.SA_y_preds = SA_y_preds
#        self_.folds_splits = folds_splits
#        self_.SA_algs = SA_algs

#        algs_combis_copy = copy.deepcopy(self_.algs_combis_dict)
#        CollectionsInstruments.delete_dict_elements_by_removal_list(self_.algs_combis_dict, list(self_.algs_combis_dict.keys()))
#        #self_.tasks = JoinableQueue()
#        m = multiprocessing.Manager() #ПРОБЛЕМА: здесь происходит зависание программы, без выброски исключения, всё остальное в теории должно работать
#        lock = m.Lock()
#        self_.tasks = m.Queue()
#        n_threads = ServiceInstruments.calc_optimal_threads_amount()
#        print('used threads:', n_threads)
#        self_.workers = [multiprocessing.Process(target=self_.validate_combis, args=(self_.tasks,lock,)) for i in range(n_threads)]
#        self_._put_tasks(algs_combis_copy)
#        self_._run_workers()
#        self_.tasks.join()
#        self_._put_stop_flags()
            
#    def _run_workers(self):
#        for w in self.workers:
#            w.start()

#    @staticmethod
#    def validate_combis(tasks_queue, lock):
#        while True:
#            task = tasks_queue.get()
#            print(task)
#            if task == None:
#                tasks_queue.task_done()
#                break
#            TrivialCombinationsValidator.validate_combi(task.combi_obj,task.SA_y_preds, task.folds_splits, task.SA_algs)
#            with lock:
#                add_combi(combi_name,combi_obj, task.multiproc_validator_obj)
#            tasks_queue.task_done()

#    @staticmethod
#    def _add_combi(combi_name, combi_obj, self_): #single threading
#        self_.algs_combis_dict[combi_name] = combi_obj

#    def _put_tasks(self, algs_combis):
#        for combi_name in algs_combis:
#            task = TrivialCombinationsValidator.TrivialCombination.TrivialCombiTask(
#                combi_name, algs_combis[combi_name],copy.deepcopy(self.SA_y_preds), copy.deepcopy(self.folds_splits), copy.deepcopy(self.SA_algs), self)
#            self.tasks.put(task)

#    def _put_stop_flags():
#        for i in range(n_threads):
#            self.tasks.put(None)
        