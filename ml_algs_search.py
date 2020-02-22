#ODC - OptimalDisjunctiveCombination
#DC - DisjunctiveCombination
#OCC - OptimalConjuctiveCombination
#CC - ConjuctiveCombination
#SC - Single combinations (length=1)
#combis - Combinations
#группа классов ищет наилучшую комбинацию алгоритмов ML.

#Данный класс написан так, чтобы с удобством можно было добавлять сюда различные комбинации алгоритмов 
#(даже если последние реализуются классами, а не просто функциями, #стекинг) и 
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

        def get_name(self):
            combi_name = ''
            separators_for_combi_names = { AlgsBestCombinationSearcher.CombinationsTypes.SINGLE: '  ',
                                        AlgsBestCombinationSearcher.CombinationsTypes.DISJUNCTIVE:' + ', 
                                          AlgsBestCombinationSearcher.CombinationsTypes.CONJUNCTIVE: ' * ' }
            separator_for_name = separators_for_combi_names[self.type]
            for alg_name in self.algs_names:
                combi_name += alg_name + separator_for_name
            return combi_name[0:-len(separator_for_name)]

    def __init__(self):
        self.__algs_SC = [] #алгоритмы по одиночке (длина комбинации = 1), важно рассматривать их отдельно для устранения дубликатов
        self.__algs_DC = [] #длина больше 1
        self.__algs_CC = [] 
        self.__folds = []

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

    def __export_results(self, algs_combis, results_from, enable_first_combis_with_unique_algs_filter = True): #results - {algs_combi_name, {metrics}}
        def switch_loggers():
            lfp = LogsFileProvider()
            if (results_from == 1):
                return lfp.ml_OCC_sorted_f1, lfp.ml_OCC_sorted_recall
            if (results_from == 2):
                return lfp.ml_ODC_sorted_f1, lfp.ml_ODC_sorted_recall
            if (results_from == 3):
                return lfp.ml_ODC_OCC_sorted_f1, lfp.ml_ODC_OCC_sorted_recall
            if (results_from == 4):
                return lfp.ml_single_algs_sorted_f1, None

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
                algs_entries_in_combis = dict(zip([single_alg.get_name() for single_alg in self.__algs_SC], [0 for _ in self.__algs_SC]))
                for algs_combi in sorted_algs_combis:
                    mark_algs_entries_in_combi()
                    if (is_combi_content_unique_alg()):
                        combis_with_unique_algs.append(algs_combi)
                return combis_with_unique_algs

            if (logger == None):
                return

            sorted_algs_combis = AlgsBestCombinationSearcher.__sort_algs_combis_by_q_metrics(algs_combis, 
                                                                                         criterias = criterias_list)
            if enable_first_combis_with_unique_algs_filter:
                sorted_algs_combis_with_unique_algs = find_first_combis_with_unique_algs_in_results()
                logger.info("//// SORTED_RESULTS_COMBIS_WITH_UNIQUE_ALGS ////")
                AlgsBestCombinationSearcher.__export_searcher_results(sorted_algs_combis_with_unique_algs, logger)
            logger.info("\n//// ALL RESULTS ////")
            AlgsBestCombinationSearcher.__export_searcher_results(sorted_algs_combis, logger)

        f1_logger, recall_logger = switch_loggers() #если logger == null, то лог просто не выведется, ошибки не будет
        export_sorted_by(f1_logger, [('f1', True), ('rec', True), ('pred_time', False)])
        export_sorted_by(recall_logger, [('rec', True), ('f1', True), ('pred_time', False)]) #можно включить

    def __test_single_algs_on_folds(self): #запоминаем результаты, данные каждым алгоритмом в отдельности, на каждом фолде
        single_algs_y_pred = {}
        single_algs_perf_q_metrics = []

        for alg in self.__algs_SC:
            train_times_on_folds = []
            pred_time_on_folds = []
            single_algs_y_pred[alg.get_name()] = []
            for (X_trainFolds, y_trainFolds, X_validFold, y_validFold) in self.__folds:
                #time.time() - показывает реальное время, а не время CPU, поэтому результаты не очень точные, но этой точности
                #достаточно, для реализации более точного подсчета с timeit нужно писать куда больше кода. Также, нам не важно
                #получить абсолютные достоверные значения, важно дифференцировать алгоритмы друг с другом.
                t0 = time.time()
                alg.algs_objs[0].learn(X_train = X_trainFolds, y_train = y_trainFolds)
                t1 = time.time()
                y_pred_alg = alg.algs_objs[0].predict(X_test = X_validFold)
                t2 = time.time()
                single_algs_y_pred[alg.get_name()].append(y_pred_alg)
                train_times_on_folds.append(t1-t0)
                pred_time_on_folds.append(t2-t1)
            single_algs_perf_q_metrics.append({'train_time': np.mean(train_times_on_folds), 'pred_time': np.mean(pred_time_on_folds)})
        single_algs_det_q_metrics = self.calc_mean_det_q_metrics_on_folds_for_algs_combis(self.__algs_SC, single_algs_y_pred)
        AlgsBestCombinationSearcher.write_metrics_to_combi_objs(self.__algs_SC, single_algs_det_q_metrics, single_algs_perf_q_metrics)
        print('////////////////// test_single_algs_on_folds() done')
        return single_algs_y_pred

    def run(self, X, y, k_folds, algs, enable_OCC = True):
        def export_odc_occ_general_results():
            odc_occ_combis = self.__algs_SC + self.__algs_DC + self.__algs_CC
            self.__export_results(odc_occ_combis, 3)

        self.__tune(X, y, k_folds, algs, enable_OCC)
        single_algs_y_preds_on_folds = self.__test_single_algs_on_folds()

        self.__run_ODC_OCC_searcher(single_algs_y_preds_on_folds, find_OCC = False)
        self.__export_results(self.__algs_SC + self.__algs_DC, 2)

        if (enable_OCC):
            self.__run_ODC_OCC_searcher(single_algs_y_preds_on_folds, find_OCC = True)
            self.__export_results(self.__algs_SC + self.__algs_CC, 1)
            export_odc_occ_general_results()

        self.__export_results(self.__algs_SC, 4, False) #вызывать до рассчетов комбинаций нельзя, 
        #ибо значения метрик будут не округленными (округление не производится сразу, чтобы метрики комбинаций рассчитывались с высокой точностью)

    #det - detection, perf - perfomance
    def __tune(self, X, y, k_folds, algs, enable_OCC, combination_length = 4, 
               det_metrics_exported_vals_decimal_places = 4, perf_metrics_exported_vals_decimal_places = 7): 
        #Сочетания без повторений (n,k) для всех k до заданного - это и есть все подмножества
        #algs НЕ должен компоноваться элементами None (нет алгоритма)
        def make_algs_combinations(algs_list):
            def init_algs_combis(combis_list, combis_type, min_length = 1, max_length = combination_length):
                for subset in algs_subsets: 
                    #в зависимости от min/max_length фильтруем список комбинаций
                    if (len(subset) >= min_length and len(subset) <= max_length):
                        algs_names, algs_objs = zip(*subset)
                        combis_list.append(AlgsBestCombinationSearcher.AlgsCombination(combis_type, algs_names, algs_objs))
                            
            algs_subsets = MathInstruments.make_subsets(algs_list, self.__combination_length+1) 
            #[ [(alg_i_name, alg_i_obj),(),()...],[],[]... ]
            init_algs_combis(self.__algs_SC, AlgsBestCombinationSearcher.CombinationsTypes.SINGLE, min_length=1, max_length=1)
            init_algs_combis(self.__algs_DC, AlgsBestCombinationSearcher.CombinationsTypes.DISJUNCTIVE, min_length=2, max_length=combination_length)
            if (enable_OCC):
                init_algs_combis(self.__algs_CC, AlgsBestCombinationSearcher.CombinationsTypes.CONJUNCTIVE, min_length=2, max_length=combination_length)
                
        self.__det_metrics_exported_vals_decimal_places = det_metrics_exported_vals_decimal_places
        self.__perf_metrics_exported_vals_decimal_places = perf_metrics_exported_vals_decimal_places
        self.__k = k_folds
        self.__X = X
        self.__y = y
        self.__combination_length = combination_length #данный параметр нужен, если алгоритм общего вида и способен расставлять
        #алгоритмы по k местам, тогда данный параметр нужно иницииализировать через prepare
        make_algs_combinations(list(algs.items())) 
        self.__folds = DatasetInstruments.make_stratified_split_on_stratified_k_folds(X,y,self.__k)
    
    def calc_mean_det_q_metrics_on_folds_for_algs_combis(self, algs_combis, y_pred_combis_on_folds):
        def calc_detection_quality_metrics_on_fold(y_pred, y_valid):
            acc = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred)
            prec = precision_score(y_valid, y_pred)
            rec = recall_score(y_valid, y_pred)
            auc = roc_auc_score(y_valid, y_pred)
            return {'f1': f1, 'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec}

        def calc_mean_metrics_vals_for_combi():
            #Раскомментировать для логирования
            #LogsFileProvider().ml_research_general.info('--------- Проверка правильности подсчёта метрик ---------')
            n = len(combi_det_q_metrics_on_folds)
            mean_det_q_metrics_for_combi = { 'f1': 0, 'auc': 0, 'acc': 0, 'prec': 0, 'rec': 0}
            for metrics in combi_det_q_metrics_on_folds:
                #Раскомментировать для логирования
                #LogsFileProvider().ml_research_general.info(metrics)
                metrics_on_fold = metrics.values()
                mean_det_q_metrics_for_combi = { metric_name:metric_val+new_metric_val/n for (metric_name, metric_val),new_metric_val in 
                                                zip(mean_det_q_metrics_for_combi.items(), metrics_on_fold) } 
                #среднее значение каждой метрики
            mean_det_q_metrics_for_combi = CollectionsInstruments.round_dict_vals(mean_det_q_metrics_for_combi, 
                                                                                  self.__det_metrics_exported_vals_decimal_places)
            #Раскомментировать для логирования
            #LogsFileProvider().ml_research_general.info('--- Итоговая метрика' + str(mean_det_q_metrics_for_combi))
            return mean_det_q_metrics_for_combi

        mean_det_q_metrics_for_combis = []
        for combi in algs_combis:
            combi_det_q_metrics_on_folds = []
            for (i,(_, _, _, y_validFold)) in enumerate(self.__folds):
                combi_det_q_metrics_on_folds.append(calc_detection_quality_metrics_on_fold(y_pred_combis_on_folds[combi.get_name()][i], y_validFold))
            mean_det_q_metrics_for_combis.append(calc_mean_metrics_vals_for_combi())
        return mean_det_q_metrics_for_combis

    @staticmethod
    def write_metrics_to_combi_objs(algs_combinations, det_quality_metrics, perf_quality_metrics): #[{det_metrics},..,{}] + [{perf_metrics},..,{}]
        for i, combi in enumerate(algs_combinations):
            combi.quality_metrics = CollectionsInstruments.merge_dicts(det_quality_metrics[i], perf_quality_metrics[i])

    def calc_perf_q_metrics_for_combis(self, algs_combinations, calc_mode = 1): 
        #предполагается, что если логика рассчета меняется от вызова к вызову, сюда можно передавать
        #параметр, который будет управлять внутренними методами (переключать их) для обеспечения требуемой логики рассчета метрик
        def calc_perf_q_metrics_for_combi_as_sum_by_algs(algs_combination): #{algs_combi_name:{'train_time':, 'pred_time': }}
            #pred_time, train_time вычисляются для комбинации подсчетом суммы значений этих метрик для каждого алгоритма в отдельности,
            #не учитывается время, которое тратится на агрегацию прогнозов (or, and функции, например), но это и не важно
            dicts = [single_algs_perf_metrics[alg_name] for alg_name in algs_combination.algs_names]
            #print(dicts)
            return CollectionsInstruments.sum_vals_of_similar_dicts(dicts, self.__perf_metrics_exported_vals_decimal_places)

        def roundOff_perf_metrics_of_SC_algs():
            for alg in self.__algs_SC:
                alg.quality_metrics['train_time'] = round(alg.quality_metrics['train_time'], self.__perf_metrics_exported_vals_decimal_places)
                alg.quality_metrics['pred_time'] = round(alg.quality_metrics['pred_time'], self.__perf_metrics_exported_vals_decimal_places)
        #для быстрого обращения к нужным данным
        single_algs_perf_metrics = {alg.get_name():{'train_time': alg.quality_metrics['train_time'], 
                                                        'pred_time': alg.quality_metrics['pred_time']} for alg in self.__algs_SC}
        combis_perf_metrics = []
        if (calc_mode == 1):
            calc_perf_q_metrics_func = calc_perf_q_metrics_for_combi_as_sum_by_algs

        for combi in algs_combinations:
            combis_perf_metrics.append(calc_perf_q_metrics_func(combi))

        roundOff_perf_metrics_of_SC_algs() #мы посчитали метрики производительности комбинаций сложением с высокой точностью на основе 
        #значений этих метрик в SC_algs, теперь
        #их надо тоже округлить, чтобы обеспечить удобство просмотра результатов

        return combis_perf_metrics

    def __run_ODC_OCC_searcher(self, single_algs_y_preds, find_OCC = False):
        def calc_y_preds_combi_on_folds():
            y_pred_combis_on_folds = {}

            combis_aggregation_func = np.logical_and if find_OCC else np.logical_or
            y_pred_combi_init_func = np.ones if find_OCC else np.zeros

            for combi in algs_combinations:
            #для обнаружения спама необходимо, чтобы хотя бы 1 алгоритм признал семпл спамом
            #фиксиоуем тренировочные фолды и валидационный и каждый алгоритм комбинации проверяем на них 
                #Раскомментировать для логирования
                #LogsFileProvider().ml_research_general.info('---------' + str(self.get_algs_combination_name(combi)))
                y_pred_combis_on_folds[combi.get_name()] = []
                for (i,(_, _, _, y_validFold)) in enumerate(self.__folds):
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
                    y_pred_combis_on_folds[combi.get_name()].append(y_pred_combination)
            return y_pred_combis_on_folds

        algs_combinations = self.__algs_CC if find_OCC else self.__algs_DC
        y_pred_combis_on_folds = calc_y_preds_combi_on_folds() #предсказания комбинаций на фолдах
        combis_det_quality_metrics = self.calc_mean_det_q_metrics_on_folds_for_algs_combis(
            algs_combinations, y_pred_combis_on_folds)
        combis_perf_quality_metrics = self.calc_perf_q_metrics_for_combis(algs_combinations)
        AlgsBestCombinationSearcher.write_metrics_to_combi_objs(algs_combinations, combis_det_quality_metrics, combis_perf_quality_metrics)
        print('////////////// calc_combis_quality_metrics() done')
        #algs_combis_with_q_metrics = dict(zip(self.__get_algs_combinations_names(find_OCC), algs_combis_quality_metrics))
        print('////////////////// ODC_OCC Searcher done')
        #return algs_combis_with_q_metrics
