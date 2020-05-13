#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import shutil
from os import listdir
from pathlib import Path

from datasets_preprocessing import *
from algs import *
from logs import *
from feature_extraction import *
from ml_algs_validation import *
from generic import *

#bow - bag of words

def set_libs_settings():
    ax = sns.set(style="darkgrid")

def run_single_algs_test():
    for alg in algs_for_bagging_combis.items():
        y_pred = alg[1].learn_predict(X_train, X_test, y_train)
        print(alg[0],': ', accuracy_score(y_test, y_pred))
    print('//////////////////////////// single alg test done')

def run_algs_validation():
    def run_searcher_on_dataset(X,y, k_folds, max_combinations_lengths = max_combis_lengths_dict):
        AlgsCombinationsValidator.run(X, y, k_folds, algs_dicts, enabled_combinations_types, max_combinations_lengths)
        print('//////////////////////////// algs validation done')

    def plot_classes_counts(y):
        sns.countplot(y=y)

    def visualize_samples_length(dataset_bow,y):
        def print_statistics_about_lengths_for_class(samples_lengths, class_name = 'unknown class'):
            print("-- Статистики для класса:", class_name)
            print('---- max:', np.max(samples_lengths))
            print('---- min:', np.min(samples_lengths))
            print('---- mean:', round(np.mean(samples_lengths)))
            print('---- median:', np.median(samples_lengths))

        def plot_lengths_distribution(samples_lengths, binwidth, class_name = 'unknown class'): #binwidth - ширина сегмента(палки) гистограммы
            #1 способ
            #ax = sns.distplot(samples_lengths, bins = int(180/binwidth), hist=True, kde=False, hist_kws={'edgecolor':'black'})

            #2 способ
            plt.hist(samples_lengths, bins = int(180/binwidth))
            plt.title('Samples lengths distribution, class: ' + class_name)
            plt.xlabel('Sample length')
            plt.ylabel('Count')

            plt.show()

        print("Длины семплов:")
        spam_samples_lengths = [len(sample_words.split()) for i,sample_words in enumerate(dataset_bow) if y[i] == 1]
        ham_samples_lengths = [len(sample_words.split()) for i,sample_words in enumerate(dataset_bow) if y[i] == 0]
        print_statistics_about_lengths_for_class(spam_samples_lengths, 'spam')
        print_statistics_about_lengths_for_class(ham_samples_lengths, 'ham')
        plot_lengths_distribution(spam_samples_lengths, binwidth = 2, class_name='spam')
        plot_lengths_distribution(ham_samples_lengths, binwidth = 2, class_name='ham')

    def print_dataset_properties():
        print("Соотношение классов:")
        print(DatasetInstruments.calc_classes_ratio(y))

        print('samples x features: ', X.shape)
    
    def move_results_to_specific_dir():
        dir_A_path = LogsFileProvider.LOGS_DIR
        dir_B_name = test_name
        dir_B_path = str(Path(dir_A_path).parent / dir_B_name) + "\\"
        logs_files = listdir(dir_A_path)
        for file_name in logs_files:
            Path(dir_B_path).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dir_A_path + file_name, dir_B_path + file_name)
    
    for test_name, ( (dataset_bow,y), (extractor_func, extractor_params), research_params ) in test_scenarios.items():
        print('/////////////////////////////////////' + test_name)
        print('//////////////////////////// preprocessing done')
        #visualize_samples_length(dataset_bow,y)
        X = extractor_func(dataset_bow, **extractor_params) #dataset_bow -> X
        print('//////////////////////////// feature extraction done')
        #print_dataset_properties()
        
        #graph_countplot(y)
        run_searcher_on_dataset(X, y, k_folds=10, **research_params)
        move_results_to_specific_dir() #они уже экспортированы, но лежат в общем каталоге по умолчанию, их нужно перенести
        LogsFileProvider.delete_log_files_on_hot()
    LogsFileProvider.shutdown()


#Работа с любым датасетом - это прохождение необходимых этапов: 1) предобработка, 2) извлечение признаков, 3) тестирование (типы проверяемых комбинаций указываются вне сценариев),
#поэтому и созданы "тестовые сценарии", любой сценарий здесь можно задать или легко модифицировать старые. 
#Сценарии можно отключать простым комментированием. Эти сценарии можно сделать совместимыми с любым исследовательским кодом.
#{ scenarios_name: ( (dataset_bow,y), (extractor,{params}), {research_params} ) }
#препроцессинг вынесен в отдельный список по разным причинам, в том числе из-за необходимости делать тестовые сценарии с комбинациями датасетов,
#удобно перемешивать датасеты
kagle2017_preproc1 = Kagle2017DatasetPreprocessors().preprocessor_1()
enron_preproc1 = EnronDatasetPreprocessors().preprocessor_1()
kagle2016_preproc1 = KagleSMS2016DatasetPreprocessors().preprocessor_1()

#название сценария будет использовано для названия каталога логов
#данные сценарии тестируют те типы комбинаций, которые включены в переменных ниже, сценарии же сами по себе не определяют это
test_scenarios = {
    #'K2017_Email pr1 Tfidf1(ngram=(1,1))': #DONE
    #( kagle2017_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ), #( (dataset_bow,y), (extractor_func, extractor_params), research_params )
    #'K2017_Email pr1 Tfidf1(ngram=(1,2))': #для доказательства, что лучше ngram=(1,2), чем (1,1)
    #( kagle2017_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,2)}), {} ), #DONE
    #'K2017_Email pr1 Tf1(ngram=(1,2))': #для доказательства, что tfidf1 лучше при тех же n-граммах
    #( kagle2017_preproc1, (FeatureExtractors.extractor_tf_1, {'ngram_range':(1,2)}), {} ), #DONE
    #'K2017_Email pr1 Counts1': #для доказательства, что tf1 и tfidf1 лучше #DONE
    #( kagle2017_preproc1, (FeatureExtractors.extractor_words_counts_1, {}), {} ),
    'E_Email pr1 Tfidf1(ngram=(1,1))': #DONE
    ( enron_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ),
    'E_Email pr1 Tfidf1(ngram=(1,2))': #DONE
    ( enron_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,2)}), {} ),
    'K2016_SMS pr1 Tfidf1(ngram=(1,1))': #DONE
    ( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ),
    'K2016_SMS pr1 Tfidf1(ngram=(1,2))': #DONE
    ( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,2)}), {} ),
    'K2016_SMS pr1 Tfidf1(ngram=(1,3))': #проверка перспективности экстрактора с такими же параметрами для улучшения результатов на СМС, где они низкие #DONE
    ( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,3)}), {} ),
    'K2016_SMS pr1 Tfidf1(ngram=(2,2))': #проверка перспективности экстрактора с такими же параметрами для улучшения результатов на СМС, где они низкие #DONE
    ( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(2,2)}), {} ),
    }

#в валидатор для тривиальных надо передавать список алгоритмов, включащий используемые в др. типах комбинаций (т.е. текущий список - надмножество),
#чтобы фильтрация результатов полноценно работала
#для всех типов комбинаций, ведь тривиальным списком пользуются алгоритмы-одиночки
algs_for_trivial = {
        #'ComplementNB_Default': ComplementNBAlg_Default(),
        'SGDClf_Default': SGDAlg_Default(),
        #'SGDAlg_AdaptiveIters': SGDAlg_AdaptiveIters(),
        'SGDAlg_LogLoss': SGDAlg_LogLoss(),
        'ASGDAlg_Default': ASGDAlg_Default(),
        'LinearSVC_Default': LinearSVCAlg_Default(),
        'LinearSVC_Balanced': LinearSVCAlg_Balanced(),
        'LinearSVCAlg_MoreSupports': LinearSVCAlg_MoreSupports(),
        'SVCAlg_RBF_Default': SVCAlg_RBF_Default(),
        'PAA_I_Default': PAA_I_Default(),
        'PAA_II_Default': PAA_II_Default(),
        'PAA_II_Balanced': PAA_II_Balanced(),
        #'kNN_Default': KNeighborsAlg_Default(),
        #'RandomForest_Default': RandomForestAlg_Default(),
        #'RandomForest_Big': RandomForestAlg_Big(),
        #'RandomForest_Medium': RandomForestAlg_Medium(),
        #'RandomForest_Small': RandomForestAlg_Small(),
        #'RandomForest_MDepth20': RandomForestAlg_MDepth20(),
        #'RandomForest_MDepth30': RandomForestAlg_MDepth30(),
        #'RandomForest_BigBootstrap75': RandomForestAlg_BigBootstrap75(),
        #'RandomForest_Bootstrap90': RandomForestAlg_Bootstrap90(),
        #'RandomForest_Balanced': RandomForestAlg_Balanced(),
        #'Perceptron_Default': PerceptronAlg_Default()
        }

algs_for_MC_BAGC = {
        'SGDClf_Default': SGDAlg_Default(),
        'SGDAlg_LogLoss': SGDAlg_LogLoss(),
        'ASGDAlg_Default': ASGDAlg_Default(),
        'LinearSVC_Default': LinearSVCAlg_Default(),
        'LinearSVC_Balanced': LinearSVCAlg_Balanced(),
        'LinearSVCAlg_MoreSupports': LinearSVCAlg_MoreSupports(),
        'SVCAlg_RBF_Default': SVCAlg_RBF_Default(),
        'PAA_I_Default': PAA_I_Default(),
        'PAA_II_Default': PAA_II_Default(),
        'PAA_II_Balanced': PAA_II_Balanced()
        }

enabled_combinations_types = { #single algs (SA) validation включено по умолчанию
    'DC': False,
    'CC': False,
    'MC': False,
    'BAGC': True,
    'BOOSTC': False,
    'STACKC': False
    }

algs_dicts = {
    'SA': algs_for_trivial,
    'DC': algs_for_trivial,
    'CC': algs_for_trivial,
    'MC': algs_for_MC_BAGC,
    'BAGC': algs_for_MC_BAGC,
    'BOOSTC': None,
    'STACKC': None
    }

max_combis_lengths_dict = {
    'DC': 4,
    'CC': 4,
    'MC': 4,
    'BAGC': 10,
    'BOOSTC': 10
    }
 #__main__ - это скрипт, который запустился на исполнение ОС, потом он мог запустить какие угодно модули.
#подпроцессы исполняют текущий модуль каждый раз заново и не только тот, который нужен, а начинает именно с __main__, хотя он им может быть по факту не нужен
#(функции target в нем нет)
#необходимо, чтобы подпроцессы не выполняли этот блок кода, поскольку это точка входа - подпроцесс заново начнёт исполнять программу.
#похоже подпроцессы делают __name__ != __main__
if __name__ == '__main__':
    set_libs_settings()
    run_algs_validation()
    #run_single_algs_test()