#ODC - OptimalDisjunctiveCombination
#DC - DisjunctiveCombination
#OCC - OptimalConjuctiveCombination
#CC - ConjuctiveCombination
#SC - Single combinations (length=1)
#combis - Algs Combinations
#группа классов ищет наилучшую комбинацию алгоритмов ML.

#Данный класс написан так, чтобы с удобством можно было добавлять сюда различные комбинации алгоритмов 
#(даже если последние реализуются классами, а не просто функциями, #стекинг; соотв. классы можно реализовать как внутренние 
#для AlgsBestCombinationSearcher) и 
#интегрировать их в поиск лучших комбинаций
#Общий алгоритм работы:
#1)генерируем фолды
#2)тестируем на фолдах алгоритмы по-одиночке: 
#    а)запоминаем y_pred на каждом фолде, 
#    б)считаем и запоминаем средние по фолдам метрики производительности, 
#    в)считаем и запоминаем средние по фолдам метрики качества обнаружения (вызываеттся общий для всех типов комбинаций метод),
#    г)записываем все метрики в поля объектов выбранной комбинации.
#3) тестируем на фолдах комбинации**:
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
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from enum import Enum
import time
import copy

from logs import *
from generic import *

class AlgsBestCombinationSearcher(object):
    class CombinationsTypes(Enum):
        SINGLE = 0 #алгоритмы тоже считаются комбинациями, просто длинной 1
        DISJUNCTIVE = 1
        CONJUNCTIVE = 2

    class AlgsCombination(object):
        def __init__(self, combi_type, algs_names, algs_objs):
            self.type = combi_type
            self.algs_names = algs_names #[]
            self.algs_objs = algs_objs #[]
            self.quality_metrics = None #{}
            self.length = len(algs_names)

        @staticmethod
        def create_combi_name(combi_type, combi_algs_names):
            SEPARATORS_FOR_COMBI_NAMES = {AlgsBestCombinationSearcher.CombinationsTypes.SINGLE: '  ',
                                AlgsBestCombinationSearcher.CombinationsTypes.DISJUNCTIVE:' + ', 
                                    AlgsBestCombinationSearcher.CombinationsTypes.CONJUNCTIVE: ' * ' }
            combi_name = ''
            separator_for_name = SEPARATORS_FOR_COMBI_NAMES[combi_type]
            for alg_name in combi_algs_names:
                combi_name += alg_name + separator_for_name
            return combi_name[0:-len(separator_for_name)]

        def create_name(self):
            return self.create_combi_name(self.type, self.algs_names)

        

    __DETECTION_METRICS = ['f1', 'auc', 'acc', 'prec', 'rec'] #единые для всех комбинаций метрики, вынесены в отдельное место
    #для упрощения добавлений/удалений метрик из вычислений, унификации и наглядности
    __PERFOMANCE_METRICS = ['train_time', 'pred_time']

    def __init__(self):
        self.__algs_SC = {} #алгоритмы по одиночке (длина комбинации = 1), важно рассматривать их отдельно для устранения дубликатов
        self.__algs_DC = {} #длина больше 1
        self.__algs_CC = {} 
        self.__folds = []

    @staticmethod
    def __print_searcher_results(algs_combis_dict_items): #предполагается, что все типы комбинаций имеют одинаковый вид результатов
        str_ = ''
        for combi_name,combi_obj in algs_combis_dict_items:
            str_ += '---' + combi_name + '\n'
            str_ += str([metric_name + '=' + str(metric_val) for metric_name,metric_val in combi_obj.quality_metrics.items()]) + '\n'
        return str_

    @staticmethod
    def __sort_algs_combis_by_q_metrics(algs_combis_dict_items, criterias = [('f1', True)]):
        def multisort(dict_items):
            for key, enable_reverse in reversed(criterias):
                #dict_.sort(key=lambda el: el.quality_metrics[key], reverse=enable_reverse)
                dict_items = sorted(dict_items, key=lambda el: el[1].quality_metrics[key], reverse=enable_reverse)
            return dict_items

        return  multisort(algs_combis_dict_items)

    def __export_results(self, algs_combis_dict, results_from, enable_combis_with_bad_metrics_vals_remover,
            enable_useless_combis_remover,enable_excessively_long_combis_remover,
            enable_best_combis_with_unique_algs_filter = True): 
        
        #enable_best_combis_with_unique_algs_filter = True, если сортировка только по убыванию качества

        #results - {algs_combi_name, {metrics}}
        def switch_loggers():
            lfp = LogsFileProvider()
            if (results_from == 1):
                return lfp.loggers['ml_OCC_sorted_f1'], lfp.loggers['ml_OCC_sorted_recall']
            if (results_from == 2):
                return lfp.loggers['ml_ODC_sorted_f1'], lfp.loggers['ml_ODC_sorted_recall']
            if (results_from == 3):
                return lfp.loggers['ml_ODC_OCC_sorted_f1'], lfp.loggers['ml_ODC_OCC_sorted_recall']
            if (results_from == 4):
                return lfp.loggers['ml_single_algs_sorted_f1'], None

        def export_sorted_by(logger, criterias_list):
            def log_filtered_results(filter_func, filter_params={}, log_header=LogsFileProvider.LOG_CONTENT_UNKN_HEADER):
                #param-ы фильтра в упакованном виде именно для вывода в лог, а так можно было обойтись без упаковок/распаковок
                filtered_results = filter_func(**filter_params)
                LogsFileProvider.log_named_info_block(logger, self.__print_searcher_results(filtered_results), 
                                                      filter_params, log_header)

            def log_with_filters():
                def find_best_unique_algs_combis_in_results(entries_amount_of_alg = 1): 
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
                            if algs_entries_in_combis[alg_name] == entries_amount_of_alg:
                                return True
                        return False

                    combis_with_unique_algs = []
                    algs_entries_in_combis = dict(zip([single_alg_name for single_alg_name in self.__algs_SC], [0 for _ in self.__algs_SC]))
                    for combi_name, combi_obj in sorted_algs_combis_di:
                        mark_algs_entries_in_combi()
                        if (is_combi_content_unique_alg()):
                            combis_with_unique_algs.append((combi_name,combi_obj))
                    return combis_with_unique_algs

                #все фильтры работают по следующей логике: перебираются отсортированные данные, если элемент данных удовлетворяет
                #условию, он добавляется в структуру данных, которая потом и будет выводиться
                if enable_best_combis_with_unique_algs_filter:
                    log_filtered_results(find_best_unique_algs_combis_in_results, 
                                         {'entries_amount_of_alg':1}, "//// BEST UNIQUE COMBINATIONS ////")
            
                    
            if (logger == None):
                return

            logger.info("Sorted by: " + str(criterias_list))
            logger.info("Useless combinations removed: " + str(enable_useless_combis_remover))
            logger.info("Сombis with bad metrics removed: " + str(enable_combis_with_bad_metrics_vals_remover))
            logger.info("Excessively long combinations removed: " + str(enable_excessively_long_combis_remover))
            logger.info("(These removers do not work for all types of combinations (despite True))")
            logger.info("") #enter

            #далее рассматривается dict как dict_items, т.к. для сортировки нужно задать новый порядок - его невозможно
            #задать в словаре, нужен другой его вид - список, не являющийся копией словаря. Даже если сортировка не будет производиться,
            #весь код далее для универсальности необходимо оставить адаптированным к dict_items
            #Ремуверам ни к чему работать в таком виде, потому что для них не имеет значения порядок комбинаций, 
            #фильтры - другое дело.
            algs_combis_dict_items = algs_combis_dict.items()
            sorted_algs_combis_di = AlgsBestCombinationSearcher.__sort_algs_combis_by_q_metrics(algs_combis_dict_items, 
                                                                                criterias = criterias_list)
            
            log_with_filters()
            LogsFileProvider.log_named_info_block(logger, AlgsBestCombinationSearcher.__print_searcher_results(sorted_algs_combis_di), 
                                      log_header="//// ALL COMBINATIONS ////")
            

        f1_logger, recall_logger = switch_loggers() #если logger == null, то лог просто не выведется, ошибки не будет
        export_sorted_by(f1_logger, [('f1', True), ('rec', True), ('pred_time', False)])
        export_sorted_by(recall_logger, [('rec', True), ('f1', True), ('pred_time', False)])

    def __test_single_algs_on_folds(self): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
        single_algs_y_pred = {}
        single_algs_perf_q_metrics = []

        for alg_name in self.__algs_SC:
            alg_obj = self.__algs_SC[alg_name]
            train_times_on_folds = []
            pred_time_on_folds = []
            single_algs_y_pred[alg_obj.create_name()] = []
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.__folds:
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
            single_algs_perf_q_metrics.append(CollectionsInstruments.create_dict_by_keys_and_vals(self.__PERFOMANCE_METRICS,
                                                                                [np.mean(train_times_on_folds), np.mean(pred_time_on_folds)]))
        single_algs_det_q_metrics = self.calc_mean_det_q_metrics_on_folds_for_algs_combis(self.__algs_SC, single_algs_y_pred)
        AlgsBestCombinationSearcher.write_metrics_to_combi_objs(self.__algs_SC, single_algs_det_q_metrics, single_algs_perf_q_metrics)
        print('////////////////// test_single_algs_on_folds() done')
        return single_algs_y_pred

    def run(self, X, y, k_folds, algs, max_combination_length = 4, enable_OCC = True):
        def remove_bad_combis(combis_dict_filtered, combis_dict_full,
            enable_combis_with_bad_metrics_vals_remover,
            enable_useless_combis_remover,
            enable_excessively_long_combis_remover):
            #combis_dict_full передаётся для аналитики при фильтрации, если пытаться определять его по комбинациям combis_dict_filtered,
            #то код сломается при внедрении новых типов комбинаций.

            #Бесполезные комбинации - это такие, которые не превосходят в одной из 
            #основных метрик (recall, precision, pred_time*) все алгоритмы-участники по одиночке
            #*по факту не проверяется, поскольку комбинация всегда работает медленнее её участников по-одиночке,
            #поэтому если комбинация не лучше в prec или rec, то она считается бесполезной
            #если требуется уточнение "на сколько лучше", то менять необходимо кол-во знаков после запятой
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
                single_algs_metrics = [copy.deepcopy(self.__algs_SC[alg_name].quality_metrics)            
                                        for alg_name in combi.algs_names] #алгоритмы участники комбинации

                for alg_metrics in single_algs_metrics:
                    CollectionsInstruments.round_dict_vals(alg_metrics, 
                                                           self.__det_metrics_exported_vals_decimal_places, self.__DETECTION_METRICS)
                    CollectionsInstruments.round_dict_vals(alg_metrics, 
                                                           self.__perf_metrics_exported_vals_decimal_places, self.__PERFOMANCE_METRICS)
                return is_combi_metrics_not_better(combi.quality_metrics, single_algs_metrics)

            #комбинации избыточной длинны - добавление нового алгоритма не улучшает никакие основные метрики
            def delete_excessively_long_combinations(comparing_metrics=['rec','prec']):
                #если к комбинации прикрепить ещё один и более алгоритмов, то комбинация останется избыточной,
                #т.е. удалять такие комбинации надо в идеале комплексно - с удалением подобных.
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
                    combis_names_generated = [self.AlgsCombination.create_combi_name(parent_combi_obj.type, parent_combi_obj.algs_names + [name]) 
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
                all_algs_names = set([alg_name for alg_name in self.__algs_SC])
                for combi_name in combis_dict_full:
                    keys_removal_list = []
                    if not combi_name in combis_dict_filtered:
                        continue
                    combi_obj = combis_dict_filtered[combi_name]
                    if combi_obj.type == self.CombinationsTypes.SINGLE:
                        continue
                    combi_predecessor_name = self.AlgsCombination.create_combi_name(combi_obj.type, combi_obj.algs_names[:-1])
                    if not combi_predecessor_name in combis_dict_filtered:
                        continue
                    combi_predecessor = combis_dict_filtered[combi_predecessor_name] if len(combi_obj.algs_names[:-1]) > 1 else self.__algs_SC[combi_predecessor_name]
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

            def is_combi_has_bad_metrics_vals(combi, not_bad_metr_vals_range={'prec':[0.9,1.0], 'rec':[0.85,1.0]}):
                #фильтрация по диапазонам метрик качества обнаружения
                #можно фильтровать по разным метрикам, не только указанным
                def is_combi_has_bad_metric_val(metric_name, vals_range):
                    return not (vals_range[0] <= combi.quality_metrics[metric_name] <= vals_range[1])
                
                for metric_name, vals_range in not_bad_metr_vals_range.items(): #(metric_name, [])
                    if is_combi_has_bad_metric_val(metric_name, vals_range):
                        return True
                return False

            #все ремуверы работают по единой схеме: идёт перебор комбинаций и удаляются те, которые не отвечают требованиям
            #наличие single_combis_compatibility не является избыточной мерой, каждый ремувер характеризуется совместимостью и не важно 
            #будет или нет внешний код сам понимать когда нужно выключать какие-либо фильтры во избежание проблем с совместимостью
            def delete_combis_by_condition(condition_func, single_combis_compatibility, params):
                #Раскомментировать для логирования
                #LogsFileProvider().loggers['ml_research_calculations'].info("///// "+str(condition_func))

                keys_removal_list=[]
                for combi_name in combis_dict_filtered:
                    if not single_combis_compatibility:
                        if combis_dict_filtered[combi_name].type == self.CombinationsTypes.SINGLE:
                            continue
                    if condition_func(combis_dict_filtered[combi_name], **params):
                        #Раскомментировать для логирования
                        #LogsFileProvider().loggers['ml_research_calculations'].info(combi_name)
                        keys_removal_list.append(combi_name)
                CollectionsInstruments.delete_dict_elements_by_removal_list(combis_dict_filtered, keys_removal_list)

            if enable_useless_combis_remover:
                delete_combis_by_condition(is_combi_useless, False, {})
            if enable_excessively_long_combis_remover:
                delete_excessively_long_combinations()
            if enable_combis_with_bad_metrics_vals_remover:
                delete_combis_by_condition(is_combi_has_bad_metrics_vals, True, {})

        def postprocess_and_export_results():
            def export_odc_occ_general_results():
                odc_occ_combis = {**algs_SC,**algs_DC, **algs_CC}
                self.__export_results(odc_occ_combis, 3, **enabled_removers)

            def roundOff_metrics_of_combis(algs_combis_dict):
                for combi_name in algs_combis_dict:
                    CollectionsInstruments.round_dict_vals(algs_combis_dict[combi_name].quality_metrics, 
                                                           self.__det_metrics_exported_vals_decimal_places, self.__DETECTION_METRICS)
                    CollectionsInstruments.round_dict_vals(algs_combis_dict[combi_name].quality_metrics, 
                                                           self.__perf_metrics_exported_vals_decimal_places, self.__PERFOMANCE_METRICS)

            #Для экспорта используются копии объектов, поскольку исследовательские данные и данные для экспорта, особенно при использовании
            #фильтров, парой сильно отличаются, но для той же фильтрации нужны именно исходные, т.е. исследовательские данные - их
            #нельзя менять или удалять. К тому же, должна быть возможность всегда использовать именно исследовательские данные,
            #а не адаптированные версии под экспорт, даже после самой процедуры экспорта, это делает код менее зависимым от очередности.
            algs_SC, algs_DC, algs_CC = copy.deepcopy(self.__algs_SC), copy.deepcopy(self.__algs_DC), copy.deepcopy(self.__algs_CC)
            #фильтрация (фильтры и ремуверы) производится на основе округлённых значений, нет смысла разделять работу фильтрации и 
            #печати результатов
            roundOff_metrics_of_combis(algs_SC)
            roundOff_metrics_of_combis(algs_DC)
            roundOff_metrics_of_combis(algs_CC)

            #фильтры - формируют подмножество данных, ремуверы - удаляют определённые результаты. Использование обоих инструментов называется фильтрацией.
            remove_bad_combis(algs_SC, self.__algs_SC, **enabled_removers)
            remove_bad_combis(algs_DC, self.__algs_DC, **enabled_removers) 
            self.__export_results({**algs_SC,**algs_DC}, 2, **enabled_removers)
            if (enable_OCC):
                remove_bad_combis(algs_CC, self.__algs_CC, **enabled_removers)
                self.__export_results({**algs_SC, **algs_CC}, 1, **enabled_removers)
                export_odc_occ_general_results()
            
            self.__export_results(algs_SC, 4, **enabled_removers, enable_best_combis_with_unique_algs_filter = False)
        
        enabled_removers = {'enable_combis_with_bad_metrics_vals_remover': True,
            'enable_useless_combis_remover': True,
            'enable_excessively_long_combis_remover': True} #параметры вынесены из-за того, что фильтры и ремуверы отделены друг от друга и 
        #в логи тоже необходимо сообщить какие параметры использовались при ремувинге

        self.__tune(X, y, k_folds, algs, enable_OCC, max_combination_length)
        single_algs_y_preds_on_folds = self.__test_single_algs_on_folds()
        self.__run_ODC_OCC_searcher(single_algs_y_preds_on_folds, find_OCC = False)
        if (enable_OCC):
            self.__run_ODC_OCC_searcher(single_algs_y_preds_on_folds, find_OCC = True) 
        
        postprocess_and_export_results() #после проведения всех исследований производится экспорт результатов

    #det - detection, perf - perfomance
    def __tune(self, X, y, k_folds, algs, enable_OCC, max_combination_length, 
               det_metrics_exported_vals_decimal_places = 4, perf_metrics_exported_vals_decimal_places = 7): 
        #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        def make_algs_combinations(algs_list):
            def init_algs_combis(combis_dict, combis_type, min_length = 1, max_length = max_combination_length):
                for subset in algs_subsets: 
                    #в зависимости от min/max_length фильтруем список комбинаций
                    if (len(subset) >= min_length and len(subset) <= max_length):
                        algs_names, algs_objs = zip(*subset)
                        algs_names, algs_objs = list(algs_names), list(algs_objs)
                        algs_combi = self.AlgsCombination(combis_type, algs_names, algs_objs)
                        combis_dict[algs_combi.create_name()] = algs_combi
                            
            algs_subsets = MathInstruments.make_subsets(algs_list, self.__max_combination_length+1)  #для остального кода важно, чтобы subset-ы
            #алгоритмов укладывались последовательно по мере увеличения длины комбинаций: сначала длин 1, потом 2...
            #[ [(alg_i_name, alg_i_obj),(),()...],[],[]... ]
            init_algs_combis(self.__algs_SC, self.CombinationsTypes.SINGLE, min_length=1, max_length=1)
            init_algs_combis(self.__algs_DC, self.CombinationsTypes.DISJUNCTIVE, min_length=2, max_length=max_combination_length)
            if (enable_OCC):
                init_algs_combis(self.__algs_CC, self.CombinationsTypes.CONJUNCTIVE, min_length=2, max_length=max_combination_length)
                
        self.__det_metrics_exported_vals_decimal_places = det_metrics_exported_vals_decimal_places
        self.__perf_metrics_exported_vals_decimal_places = perf_metrics_exported_vals_decimal_places
        self.__max_combination_length = max_combination_length #данный параметр нужен, если алгоритм общего вида и способен расставлять
        #алгоритмы по k местам, тогда данный параметр нужно иницииализировать через prepare
        make_algs_combinations(list(algs.items())) 
        self.__folds = DatasetInstruments.make_stratified_split_on_stratified_k_folds(X,y,k_folds)
    
    def calc_mean_det_q_metrics_on_folds_for_algs_combis(self, algs_combis_dict, y_pred_combis_on_folds):
        def calc_detection_quality_metrics_on_fold(y_pred, y_valid):
            acc = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred)
            prec = precision_score(y_valid, y_pred)
            rec = recall_score(y_valid, y_pred)
            auc = roc_auc_score(y_valid, y_pred)
            return CollectionsInstruments.create_dict_by_keys_and_vals(self.__DETECTION_METRICS,[f1, auc, acc, prec, rec])

        def calc_mean_metrics_vals_for_combi():
            #Раскомментировать для логирования
            #LogsFileProvider().loggers['ml_research_calculations'].info('--------- Проверка правильности подсчёта метрик ---------')
            n = len(combi_det_q_metrics_on_folds)
            mean_det_q_metrics_for_combi = CollectionsInstruments.create_dict_by_keys_and_vals(self.__DETECTION_METRICS,[0, 0, 0, 0, 0])
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
            for (i,(_, _, _, y_validFold)) in enumerate(self.__folds):
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
            return CollectionsInstruments.sum_vals_of_similar_dicts(dicts, self.__perf_metrics_exported_vals_decimal_places)

        #для быстрого обращения к нужным данным
        single_algs_perf_metrics = {alg_name:{perf_metric_name: self.__algs_SC[alg_name].quality_metrics[perf_metric_name] 
                                                        for perf_metric_name in self.__PERFOMANCE_METRICS} for alg_name in self.__algs_SC}
        combis_perf_metrics = []
        if (calc_mode == 1):
            calc_perf_q_metrics_func = calc_perf_q_metrics_for_combi_as_sum_by_algs

        for combi_name in algs_combinations:
            combis_perf_metrics.append(calc_perf_q_metrics_func(algs_combinations[combi_name]))

        return combis_perf_metrics

    def __run_ODC_OCC_searcher(self, single_algs_y_preds, find_OCC = False):
        def calc_y_preds_combi_on_folds():
            y_pred_combis_on_folds = {}

            combis_aggregation_func = np.logical_and if find_OCC else np.logical_or
            y_pred_combi_init_func = np.ones if find_OCC else np.zeros

            for combi_name in algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них 
                #Раскомментировать для логирования
                #LogsFileProvider().loggers['ml_research_calculations'].info('---------' + str(self.get_algs_combination_name(combi)))
                combi_obj = algs_combinations[combi_name]
                y_pred_combis_on_folds[combi_name] = []
                for (i,(_, _, _, y_validFold)) in enumerate(self.__folds):
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

        algs_combinations = self.__algs_CC if find_OCC else self.__algs_DC
        y_pred_combis_on_folds = calc_y_preds_combi_on_folds() #предсказания комбинаций на фолдах
        combis_det_quality_metrics = self.calc_mean_det_q_metrics_on_folds_for_algs_combis(
            algs_combinations, y_pred_combis_on_folds)
        combis_perf_quality_metrics = self.calc_perf_q_metrics_for_combis(algs_combinations)
        AlgsBestCombinationSearcher.write_metrics_to_combi_objs(algs_combinations, combis_det_quality_metrics, combis_perf_quality_metrics)
        print('////////////// calc_combis_quality_metrics() done')
        print('////////////////// ODC_OCC Searcher done')
