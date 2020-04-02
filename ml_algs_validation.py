#DC - DisjunctiveCombination
#CC - ConjuctiveCombination
#SA - Single algs (length=1)
#MC - MAJORITY 
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
#3) логгируем.
#Таким образом, какие бы комбинации не тестировались, они должны быть совместимы с основными этапами (должным образом унифицированы).
#Тривиальные комбинации связаны логической операцией, не более того. Усложнённые комбинации не просто связывают результаты работы каждого алгоритма,
#обучение каждого алгоритма каждой комбинации может проходить по-разному из-за разницы в обучающих семплах, в параметрах, доступных признаках..., а
#значит сначала запомнить предсказания алгоритмов-одиночек, а потом их связать не получится.

#Этап 2 при поиске тривиальных комбинаций:
#1)тестируем на фолдах алгоритмы по-одиночке: 
#    а)запоминаем y_pred на каждом фолде, 
#    б)считаем и запоминаем средние по фолдам метрики производительности, 
#    в)считаем и запоминаем средние по фолдам метрики качества обнаружения (вызываеттся общий для всех типов комбинаций метод),
#    г)записываем все метрики в поля объектов выбранной комбинации.
#2) тестируем на фолдах комбинации**:
#    а)используем (или заново считаем, если стекинг#) 2a* для вычисления и запоминания предсказаний комбинаций на фолдах (y_pred_combis_on_folds),
#    б)считаем и запоминаем метрики качества обнаружения (вызываеттся общий для всех типов комбинаций метод),
#    в)считаем метрики производительности (разные методы, в зависимости от типа комбинации, но может быть одинаковая логика),
#    г)записываем все метрики в поля объектов выбранной комбинации.
#*они связаны логическим законом комбинации: дизъюнкцией, конъюнкцией..., однако комбинации могут быть сложнее, чем просто один связующий закон,
#тогда придётся не полагаться на уже посчитанные на этапе 2а y_pred
#**на данном этапе вызываются методы (они должны удовлетворять пунктам 3а-3г, такая формализация позволяет лучше обобщить код) 
#для каждого типа комбинации (метод run()), 
#там же настраивается экспорт результатов.
#Почему нельзя убрать подпункт г? Потому что метрики собираются разными способами, их невозможно сохранить сразу в одном месте, 
#а хранить в одном виде их нужно для удобства сортировки.

#Проводить тестирование single_algs необходимо независимо от того исследуются комбинации или нет, потому что данные о работе алгоритмов-одиночек используются
#при фильтрации любых типов комбинаций

#Дочерние классы валидаторов не сделаны статическими, поскольку должна быть возможность одновременной работы с разными экземплярами исследований.

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from enum import Enum
import time
import copy
from abc import ABC, abstractmethod, abstractstaticmethod

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
    _folds = None
    _det_metrics_exported_vals_decimal_places = 4
    _perf_metrics_exported_vals_decimal_places = 7
    _max_combination_length = 4

    @staticmethod
    def run(X, y, k_folds, algs_dicts, enabled_combinations_types, max_combination_length = 4, 
            det_metrics_exported_vals_decimal_places = 4, perf_metrics_exported_vals_decimal_places = 7): #совмещение исследований всех валидаторов
        AlgsCombinationsValidator._folds = DatasetInstruments.make_stratified_split_on_stratified_k_folds(X,y,k_folds)
        AlgsCombinationsValidator._det_metrics_exported_vals_decimal_places = det_metrics_exported_vals_decimal_places
        AlgsCombinationsValidator._perf_metrics_exported_vals_decimal_places = perf_metrics_exported_vals_decimal_places
        AlgsCombinationsValidator._max_combination_length = max_combination_length

        acv = AlgsCombinationsValidator()
        tcv = TrivialCombinationsValidator(algs_dicts['TRIVIAL'],
                                           enable_DC = enabled_combinations_types['DC'], enable_CC = enabled_combinations_types['CC'])
        tcv._validate()
        #ccv = ComplexCombinationsValidator(enabled_combinations_types)
        #ccv._validate(tcv._algs_SA)
        acv._postprocess_and_export_results(enabled_combinations_types, tcv)

    @abstractmethod
    def _init_combination(self, combis_type, algs_names, algs_objs):
        ...

    @abstractmethod
    def _make_algs_combinations(self, algs_dict_items, algs_subsets):
        ...

    @abstractmethod
    def _validate(self, **params): #запуск валидатора, но без постпроцессинга и логирования
        ...

    def _postprocess_and_export_results(self, enabled_combinations_types, tcv, ccv = None): #этапы пост-обработки и экспорта объединены из-за их тесной связности
        def export_united_results_for_all_combis_types():
            self._export_results(all_types_combis_dict, 3, tcv._algs_SA, enabled_removers, enabled_highlighters)

        enabled_removers = {'combis_with_bad_metrics_vals_remover': True,
        'useless_combis_remover': True,
        'excessively_long_combis_remover': True} #параметры вынесены из-за того, что фильтры и ремуверы отделены друг от друга и 
        #в логи тоже необходимо сообщить какие параметры использовались при ремувинге
        enabled_removers_SA = {'combis_with_bad_metrics_vals_remover': False,
        'useless_combis_remover': False,
        'excessively_long_combis_remover': False} #отдельные тумблеры для алгоритмов-одиночек нужны, поскольку нужно оценить их в отдельном порядке
        #например, важно знать качество созданный модификаций алгоритмов

        enabled_highlighters = { 'best_combis_with_unique_algs_filter': True }
        enabled_highlighters_SA = { 'best_combis_with_unique_algs_filter': False }
        
        all_types_combis_dict = {}

        algs_SA_filtered = tcv._postprocess_results(tcv._algs_SA, tcv._algs_SA, enabled_removers_SA)
        all_types_combis_dict.update(algs_SA_filtered)
        tcv._export_results(algs_SA_filtered, 4, tcv._algs_SA, enabled_removers_SA, enabled_highlighters_SA)
        
        if enabled_combinations_types['DC']:
            algs_DC_filtered = tcv._postprocess_results(tcv._algs_DC, tcv._algs_SA,enabled_removers)
            all_types_combis_dict.update(algs_DC_filtered)
            tcv._export_results({**algs_SA_filtered,**algs_DC_filtered}, 2, tcv._algs_SA, enabled_removers, enabled_highlighters)
        if enabled_combinations_types['CC']:
            algs_CC_filtered = tcv._postprocess_results(tcv._algs_CC, tcv._algs_SA, enabled_removers)
            all_types_combis_dict.update(algs_CC_filtered)
            tcv._export_results({**algs_SA_filtered, **algs_CC_filtered}, 1, tcv._algs_SA, enabled_removers, enabled_highlighters)
        #if enabled_combinations_types['MC']:
        #    algs_MC_filtered = ccv._postprocess_results(ccv._algs_MC, tcv._algs_SA, enabled_removers)
        #    all_types_combis_dict.update(algs_MC_filtered)
        #    ccv._export_results(algs_MC_filtered, 5, enabled_removers, enabled_highlighters)

        export_united_results_for_all_combis_types()

    #@abstractmethod
    #def _postprocess_and_export_results(self, **params): 
    #    ...

    @abstractmethod 
    def _remove_bad_combis(self, combis_dict_filtered, combis_dict_full, algs_SA, enabled_removers):
        ...

    @abstractmethod
    def _log_with_highlighters(self, logger, sorted_algs_combis_di, algs_SA, enabled_highlighters):
        if enabled_highlighters['best_combis_with_unique_algs_filter']:
            filtered_results = CombinationsFiltration.highlight_best_unique_algs_combis_in_results(algs_SA, sorted_algs_combis_di)
            LogsFileProvider.log_named_info_block(logger, AlgsCombinationsValidator._make_searcher_results_str(filtered_results), 
                                                log_header="//// BEST UNIQUE COMBINATIONS ////")

    _DETECTION_METRICS = ['f1', 'auc', 'acc', 'prec', 'rec'] #единые для всех комбинаций метрики, вынесены в отдельное место
    #для упрощения добавлений/удалений метрик из вычислений, унификации и наглядности
    _PERFOMANCE_METRICS = ['train_time', 'pred_time']

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
            return lfp.loggers['ml_MAJ_sorted_f1'], lfp.loggers['ml_MAJ_sorted_recall']

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
    
    def _postprocess_results(self, algs_combis, algs_SA, enabled_removers):
            #Для экспорта используются копии объектов, поскольку исследовательские данные и данные для экспорта, особенно при использовании
    #фильтров, парой сильно отличаются, но для той же фильтрации нужны именно исходные, т.е. исследовательские данные - их
    #нельзя менять или удалять. К тому же, должна быть возможность всегда использовать именно исследовательские данные,
    #а не адаптированные версии под экспорт, даже после самой процедуры экспорта, это делает код менее зависимым от очередности.
        algs_combis_filtered = copy.deepcopy(algs_combis)
        #фильтрация производится на основе округлённых значений, нет смысла разделять работу фильтрации и 
    #печати результатов
        self._roundOff_metrics_of_combis(algs_combis_filtered)
        self._remove_bad_combis(algs_combis_filtered, algs_combis, algs_SA, enabled_removers) 
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

    def _init_algs_combis(self, algs_subsets, combis_dict, combis_type, min_length = 1, max_length = _max_combination_length): #add in-place
        for subset in algs_subsets: 
            #в зависимости от min/max_length фильтруем список комбинаций
            if (len(subset) >= min_length and len(subset) <= max_length):
                algs_names, algs_objs = zip(*subset)
                algs_names, algs_objs = list(algs_names), list(algs_objs)
                algs_combi = self._init_combination(combis_type, algs_names, algs_objs)
                combis_dict[algs_combi.create_name()] = algs_combi

class ComplexCombinationsValidator(AlgsCombinationsValidator):
    #отличие от валидатора тривиальных, данный не предусматривает запоминание результатов предсказаний алгоритмов-одиночек и их последующее комбинирование,
    #т.к. алгоритмы-участники в сложных комбинациях от комбинации к комбинации находятся в разных условиях, а не фиксированных 
    class ComplexCombination(AlgsCombinationsValidator.AlgsCombination, ABC): #стекинг, бэггинг...
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)
        
        @abstractmethod
        def fit(self):
            ...

        @abstractmethod
        def predict(self):
            ...

    class BaggingCombination(ComplexCombination):
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)

        def fit(self):
            pass

        def predict(self):
            pass

        def create_name(self):
            pass

    class MajorityCombination(ComplexCombination):
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)

        def fit(self):
            pass

        def predict(self):
            pass

        def create_name(self):
            pass

    class BoostingCombination(ComplexCombination):
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)

        def fit(self):
            pass

        def predict(self):
            pass

        def create_name(self):
            pass

    class StackingCombination(ComplexCombination): 
        #algs_names=[[{alg_names_layer1}],[],..,[]]
        #algs_objs=[[{algs_layer1}],[],..,[]]
        #size - tuple(algs_layer1_amount,..)
        #генерящийся стекинг представляет собой слои, в каждом есть алгоритмы, логику взаимодействия слоёв задают fit() и predict(),
        #возможно ещё какие-то вспомогательные методы, которые должны быть определены дополнительно
        def __init__(self, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)

        def fit(self):
            pass

        def predict(self):
            pass

        def create_name(self):
            pass

    def __init__(self, algs_dict, **enabled_types):
        if enabled_types['MC']:
            self._algs_MC = {}
        if enabled_types['BAGC']:
            self._algs_BAGC = {}
        if enabled_types['BOOSTC']:
            self._algs_BOOSTC = {}
        if enabled_types['STACKC']:
            self._algs_STACKC = {}
        self._enabled_combis_types = enabled_types

        algs_subsets = MathInstruments.make_subsets(algs_dict.items(), max_combination_length+1)
        self._make_algs_combinations(algs_subsets) 

    def _validate(self, **params):
        pass
        #if self._enabled_combis_types['MC']:
        #    ...
        #if self._enabled_combis_types['BAGC']:
            
        #if self._enabled_combis_types['BOOSTC']:
            
        #if self._enabled_combis_types['STACKC']:
            

class TrivialCombinationsValidator(AlgsCombinationsValidator):
    class TrivialCombination(AlgsCombinationsValidator.AlgsCombination):
        class Types(Enum):
            SINGLE = 0
            DISJUNCTIVE = 1
            CONJUNCTIVE = 2
        #методы predict и fit не были имплементированы, поскольку результаты работы комбинации рассчитываются
        #исходя из комбинирования результатов работы алгоритмов-одиночек
        def __init__(self, type, algs_names, algs_objs):
            super().__init__(algs_names, algs_objs)
            self.type = type

        @staticmethod
        def create_combi_name(combi_type, combi_algs_names):
            SEPARATORS_FOR_COMBI_NAMES = { TrivialCombinationsValidator.TrivialCombination.Types.SINGLE: '  ',
                                TrivialCombinationsValidator.TrivialCombination.Types.DISJUNCTIVE: ' + ', 
                                    TrivialCombinationsValidator.TrivialCombination.Types.CONJUNCTIVE: ' * ' }
            combi_name = ''
            separator_for_name = SEPARATORS_FOR_COMBI_NAMES[combi_type]
            for alg_name in combi_algs_names:
                combi_name += alg_name + separator_for_name
            return combi_name[0:-len(separator_for_name)]

        def create_name(self):
            return self.create_combi_name(self.type, self.algs_names)

    def __init__(self, algs_dict, enable_DC = False, enable_CC = False):
        #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        
        self._algs_SA = {} #алгоритмы по одиночке (длина комбинации = 1)

        if enable_DC:
            self._algs_DC = {} #длина больше 1
        if enable_CC:
            self._algs_CC = {}

        self.__enable_DC = enable_DC
        self.__enable_CC = enable_CC

        algs_subsets = MathInstruments.make_subsets(algs_dict.items(), AlgsCombinationsValidator._max_combination_length+1)  #для остального кода важно, чтобы subset-ы
        #алгоритмов укладывались последовательно по мере увеличения длины комбинаций: сначала длин 1, потом 2...
        #[ [(alg_i_name, alg_i_obj),(),()...],[],[]... ]
        self._make_algs_combinations(algs_subsets) 

    def _init_combination(self, combis_type, algs_names, algs_objs):
        return self.TrivialCombination(combis_type, algs_names, algs_objs)

    def _make_algs_combinations(self, algs_subsets):
        self._init_algs_combis(algs_subsets, self._algs_SA, self.TrivialCombination.Types.SINGLE, min_length=1, max_length=1)
        if self.__enable_DC:
            self._init_algs_combis(algs_subsets, self._algs_DC, self.TrivialCombination.Types.DISJUNCTIVE, min_length=2, 
                                   max_length=AlgsCombinationsValidator._max_combination_length)
        if self.__enable_CC:
            self._init_algs_combis(algs_subsets, self._algs_CC, self.TrivialCombination.Types.CONJUNCTIVE, min_length=2, 
                                   max_length=AlgsCombinationsValidator._max_combination_length)

    def _validate(self, **params):
        single_algs_y_preds_on_folds = self.__test_single_algs_on_folds()
        if self.__enable_DC:
            self.__run_DC_CC_validation(single_algs_y_preds_on_folds, False)
        if self.__enable_CC:
            self.__run_DC_CC_validation(single_algs_y_preds_on_folds, True) 

    def __test_single_algs_on_folds(self): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
        single_algs_y_pred = {}
        single_algs_perf_q_metrics = []

        for alg_name in self._algs_SA:
            alg_obj = self._algs_SA[alg_name]
            train_times_on_folds = []
            pred_time_on_folds = []
            single_algs_y_pred[alg_obj.create_name()] = []
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in AlgsCombinationsValidator._folds:
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
            single_algs_perf_q_metrics.append(CollectionsInstruments.create_dict_by_keys_and_vals(self._PERFOMANCE_METRICS,
                                                                                [np.mean(train_times_on_folds), np.mean(pred_time_on_folds)]))
        single_algs_det_q_metrics = self.calc_mean_det_q_metrics_on_folds_for_algs_combis(self._algs_SA, single_algs_y_pred)
        TrivialCombinationsValidator.write_metrics_to_combi_objs(self._algs_SA, single_algs_det_q_metrics, single_algs_perf_q_metrics)
        print('////////////////// test_single_algs_on_folds() done')
        return single_algs_y_pred

    def calc_mean_det_q_metrics_on_folds_for_algs_combis(self, algs_combis_dict, y_pred_combis_on_folds):
        def calc_detection_quality_metrics_on_fold(y_pred, y_valid):
            acc = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred)
            prec = precision_score(y_valid, y_pred)
            rec = recall_score(y_valid, y_pred)
            auc = roc_auc_score(y_valid, y_pred)
            return CollectionsInstruments.create_dict_by_keys_and_vals(self._DETECTION_METRICS,[f1, auc, acc, prec, rec])

        def calc_mean_metrics_vals_for_combi():
            #Раскомментировать для логирования
            #LogsFileProvider().loggers['ml_research_calculations'].info('--------- Проверка правильности подсчёта метрик ---------')
            n = len(combi_det_q_metrics_on_folds)
            mean_det_q_metrics_for_combi = CollectionsInstruments.create_dict_by_keys_and_vals(self._DETECTION_METRICS,[0, 0, 0, 0, 0])
            for metrics in combi_det_q_metrics_on_folds:
                #Раскомментировать для логирования
                #LogsFileProvider().loggers['ml_research_calculations'].info(metrics)
                metrics_on_fold = metrics.values()
                mean_det_q_metrics_for_combi = { metric_name:metric_val+new_metric_val/n for (metric_name, metric_val),new_metric_val in 
                                                zip(mean_det_q_metrics_for_combi.items(), metrics_on_fold) } 
                #среднее значение каждой метрики
            
            #Раскомментировать для логирования
            #LogsFileProvider().loggers['ml_research_calculations'].info('--- Итоговая метрика' + str(mean_det_q_metrics_for_combi))
            return mean_det_q_metrics_for_combi

        mean_det_q_metrics_for_combis = []
        for combi_name in algs_combis_dict:
            combi_obj = algs_combis_dict[combi_name]
            combi_det_q_metrics_on_folds = []
            for (i,(_, _, _, y_validFold)) in enumerate(AlgsCombinationsValidator._folds):
                combi_det_q_metrics_on_folds.append(calc_detection_quality_metrics_on_fold(y_pred_combis_on_folds[combi_name][i], y_validFold))
            mean_det_q_metrics_for_combis.append(calc_mean_metrics_vals_for_combi())
        return mean_det_q_metrics_for_combis

    @staticmethod
    def write_metrics_to_combi_objs(algs_combinations, det_quality_metrics, perf_quality_metrics): #[{det_metrics},..,{}] + [{perf_metrics},..,{}]
        for i, combi_name in enumerate(algs_combinations):
            algs_combinations[combi_name].quality_metrics = CollectionsInstruments.merge_dicts(det_quality_metrics[i], perf_quality_metrics[i])

    def calc_perf_q_metrics_for_combis(self, algs_combinations, calc_mode = 1): 
        #предполагается, что если логика рассчета меняется от вызова к вызову, сюда можно передавать
        #параметр, который будет управлять внутренними методами (переключать их) для обеспечения требуемой логики рассчета метрик
        def calc_perf_q_metrics_for_combi_as_sum_by_algs(algs_combination): #{algs_combi_name:{'train_time':, 'pred_time': }}
            #pred_time, train_time вычисляются для комбинации подсчетом суммы значений этих метрик для каждого алгоритма в отдельности,
            #не учитывается время, которое тратится на агрегацию прогнозов (or, and функции, например), но это и не важно
            dicts = [single_algs_perf_metrics[alg_name] for alg_name in algs_combination.algs_names]
            #print(dicts)
            return CollectionsInstruments.sum_vals_of_similar_dicts(dicts, AlgsCombinationsValidator._perf_metrics_exported_vals_decimal_places)

        #для быстрого обращения к нужным данным
        single_algs_perf_metrics = {alg_name:{perf_metric_name: self._algs_SA[alg_name].quality_metrics[perf_metric_name] 
                                                        for perf_metric_name in self._PERFOMANCE_METRICS} for alg_name in self._algs_SA}
        combis_perf_metrics = []
        if (calc_mode == 1):
            calc_perf_q_metrics_func = calc_perf_q_metrics_for_combi_as_sum_by_algs

        for combi_name in algs_combinations:
            combis_perf_metrics.append(calc_perf_q_metrics_func(algs_combinations[combi_name]))

        return combis_perf_metrics

    def __run_DC_CC_validation(self, single_algs_y_preds, find_CC):
        def calc_y_preds_combi_on_folds():
            y_pred_combis_on_folds = {}

            combis_aggregation_func = np.logical_and if find_CC else np.logical_or
            y_pred_combi_init_func = np.ones if find_CC else np.zeros

            for combi_name in algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них 
                #Раскомментировать для логирования
                #LogsFileProvider().loggers['ml_research_calculations'].info('---------' + str(self.get_algs_combination_name(combi)))
                combi_obj = algs_combinations[combi_name]
                y_pred_combis_on_folds[combi_name] = []
                for (i,(_, _, _, y_validFold)) in enumerate(AlgsCombinationsValidator._folds):
                    algs_combi_det_q_metrics_on_folds = [] #список dict-ов с метриками
                    y_pred_combination = y_pred_combi_init_func(y_validFold.shape, dtype=bool)
                    for alg_name in combi_obj.algs_names:
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().loggers['ml_research_calculations'].info('y_pred_combination before' + str(dict(zip(classes.tolist(), classes_counts))))
                        y_pred_alg = single_algs_y_preds[alg_name][i]
                        y_pred_combination = combis_aggregation_func(y_pred_combination, y_pred_alg)
                        #Раскомментировать для логирования
                        #classes, classes_counts = np.unique(y_pred_combination, return_counts = True)
                        #LogsFileProvider().loggers['ml_research_calculations'].info('y_pred_combination after' + str(dict(zip(classes.tolist(), classes_counts))))
                    y_pred_combis_on_folds[combi_name].append(y_pred_combination)
            return y_pred_combis_on_folds

        algs_combinations = self._algs_CC if find_CC else self._algs_DC
        y_pred_combis_on_folds = calc_y_preds_combi_on_folds() #предсказания комбинаций на фолдах
        combis_det_quality_metrics = self.calc_mean_det_q_metrics_on_folds_for_algs_combis(algs_combinations, y_pred_combis_on_folds)
        combis_perf_quality_metrics = self.calc_perf_q_metrics_for_combis(algs_combinations)
        TrivialCombinationsValidator.write_metrics_to_combi_objs(algs_combinations, combis_det_quality_metrics, combis_perf_quality_metrics)
        print('////////////// calc_combis_quality_metrics() done')
        print('////////////////// DC_CC validator done')

    def _remove_bad_combis(self, combis_dict_filtered, combis_dict_full, algs_SA, #управляет ремуверами, здесь можно вызвать любой набор
                enabled_removers):
        #combis_dict_full передаётся для аналитики при фильтрации, если пытаться определять его по комбинациям combis_dict_filtered,
        #то код сломается при внедрении новых типов комбинаций.

        if enabled_removers['useless_combis_remover']:
            TrivialCombisFiltration.remove_useless_combis(combis_dict_filtered, algs_SA)
        if enabled_removers['excessively_long_combis_remover']:
            TrivialCombisFiltration.remove_excessively_long_combinations(combis_dict_filtered, combis_dict_full, algs_SA)
        if enabled_removers['combis_with_bad_metrics_vals_remover']:
            TrivialCombisFiltration.remove_combis_with_bad_metrics_vals(combis_dict_filtered)

    def _log_with_highlighters(self, logger, sorted_algs_combis_di, algs_SA, enabled_highlighters):
        super()._log_with_highlighters(logger, sorted_algs_combis_di, algs_SA, enabled_highlighters)

    
class CombinationsFiltration(ABC): 
    #ремуверы (removers) - удаляют бесполезное и хайлайтеры (highlighter) - выделяют подмножество комбинаций опред. свойств
    # combis_dict_filtered - словарь с комбинациями, который проходит фильтрацию и не содержит первоначальный набор добытых комбинаций

    #многие ремуверы работают по единой схеме: идёт перебор комбинаций и удаляются те, которые не отвечают требованиям
    #наличие single_combis_compatibility не является избыточной мерой, каждый ремувер характеризуется совместимостью и не важно 
    #будет или нет внешний код сам понимать когда нужно выключать какие-либо фильтры во избежание проблем с совместимостью
    @staticmethod
    def _remove_combis_by_condition(condition_func, combis_dict_filtered, single_combis_compatibility, params):
        #Раскомментировать для логирования
        #LogsFileProvider().loggers['ml_research_calculations'].info("///// "+str(condition_func))

        keys_removal_list=[]
        for combi_name in combis_dict_filtered:
            if not single_combis_compatibility:
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
    @abstractstaticmethod
    def _remove_useless_combis(combis_dict_filtered, algs_SA):
        def is_combi_useless(combi, comparing_metrics=['rec','prec']):
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

            for alg_metrics in single_algs_metrics:
                CollectionsInstruments.round_dict_vals(alg_metrics, 
                                                        AlgsCombinationsValidator._det_metrics_exported_vals_decimal_places, AlgsCombinationsValidator._DETECTION_METRICS)
                CollectionsInstruments.round_dict_vals(alg_metrics, 
                                                        AlgsCombinationsValidator._perf_metrics_exported_vals_decimal_places, AlgsCombinationsValidator._PERFOMANCE_METRICS)
            return is_combi_metrics_not_better(combi.quality_metrics, single_algs_metrics)

        CombinationsFiltration._remove_combis_by_condition(is_combi_useless, combis_dict_filtered, False, {})

    #комбинации избыточной длинны - добавление нового алгоритма не улучшает никакие основные метрики
       #если к комбинации прикрепить ещё один и более алгоритмов, то комбинация останется избыточной,
            #т.е. удалять такие комбинации надо в идеале комплексно - с удалением подобных.
    @abstractstaticmethod
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

    @abstractstaticmethod
    def _remove_combis_with_bad_metrics_vals(combis_dict_filtered):
        def is_combi_has_bad_metrics_vals(combi, not_bad_metr_vals_range={'prec':[0.9,1.0], 'rec':[0.85,1.0]}):
        #фильтрация по диапазонам метрик качества обнаружения
        #можно фильтровать по разным метрикам, не только указанным
            def is_combi_has_bad_metric_val(metric_name, vals_range):
                return not (vals_range[0] <= combi.quality_metrics[metric_name] <= vals_range[1])
                
            for metric_name, vals_range in not_bad_metr_vals_range.items(): #(metric_name, [])
                if is_combi_has_bad_metric_val(metric_name, vals_range):
                    return True
            return False
        CombinationsFiltration._remove_combis_by_condition(is_combi_has_bad_metrics_vals, combis_dict_filtered, True, {})

    @abstractstaticmethod
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
    pass

class MajorityCombisFiltration(CombinationsFiltration, ABC):
    @staticmethod
    def highlight_best_unique_algs_combis_in_results(algs_SA, sorted_combis_dict_items, entries_amount_of_each_alg = 1):
        CombinationsFiltration.highlight_best_unique_algs_combis_in_results(algs_SA, sorted_combis_dict_items, entries_amount_of_each_alg)
    #@staticmethod
    #def remove_excessively_long_combinations(algs_combis_filtered, algs_combis, algs_SA, comparing_metrics=['rec','prec']):
    #    pass
            
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


    
        