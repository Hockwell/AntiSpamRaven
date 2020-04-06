#%%
import seaborn as sns
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

def set_libs_settings():
    ax = sns.set(style="darkgrid")

def run_single_algs_test():
    for alg in algs_for_bagging_combis.items():
        y_pred = alg[1].learn_predict(X_train, X_test, y_train)
        print(alg[0],': ', accuracy_score(y_test, y_pred))
    print('//////////////////////////// single alg test done')

def run_algs_validation():
    def run_searcher_on_dataset(X,y, k_folds, max_combination_length = 4):
        AlgsCombinationsValidator.run(X, y, k_folds, algs_dicts, enabled_combinations_types, max_combination_length)
        print('//////////////////////////// algs validation done')
    
    def print_dataset_properties():
        print("Соотношение классов:")
        print(DatasetInstruments.calc_classes_ratio(y))

        print('samples x features: ', X.shape)
    
    def visualize_dataset():
        def visualize_dataset(y):
            sns.countplot(y=y)

        visualize_dataset(y)
        visualize_dataset(y_train)
        visualize_dataset(y_test)

    def move_results_to_specific_dir():
        dir_A_path = LogsFileProvider.LOGS_DIR
        dir_B_name = test_name
        dir_B_path = str(Path(dir_A_path).parent / dir_B_name) + "\\"
        logs_files = listdir(dir_A_path)
        for file_name in logs_files:
            Path(dir_B_path).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dir_A_path + file_name, dir_B_path + file_name)

    for test_name, ( (corpus,y), (extractor_func, extractor_params), research_params ) in test_scenarios.items():
        print('/////////////////////////////////////' + test_name)
        print('//////////////////////////// preprocessing done')
        X = extractor_func(corpus, **extractor_params) #corpus -> X
        print('//////////////////////////// feature extraction done')
        print_dataset_properties()
        X_train, y_train, X_test, y_test = DatasetInstruments.make_shuffle_stratified_split_on_folds(X,y, test_size = 0.25, n_splits=1)[0]
        #visualize_dataset()
        run_searcher_on_dataset(X, y, k_folds=10, **research_params)
        move_results_to_specific_dir() #они уже экспортированы, но лежат в общем каталоге по умолчанию, их нужно перенести
        LogsFileProvider.delete_log_files_on_hot()
    LogsFileProvider.shutdown()

set_libs_settings()

#Работа с любым датасетом - это прохождение необходимых этапов: 1) предобработка, 2) извлечение признаков, 3) тестирование (типы проверяемых комбинаций указываются вне сценариев),
#поэтому и созданы "тестовые сценарии", любой сценарий здесь можно задать или легко модифицировать старые. 
#Сценарии можно отключать простым комментированием. Эти сценарии можно сделать совместимыми с любым исследовательским кодом.
#{ scenarios_name: ( (corpus,y), (extractor,{params}), {research_params} ) }
#препроцессинг вынесен в отдельный список по разным причинам, в том числе из-за необходимости делать тестовые сценарии с комбинациями датасетов
kagle2017_preproc1 = Kagle2017DatasetPreprocessors().preprocessor_1()
enron_preproc1 = EnronDatasetPreprocessors().preprocessor_1()
kagle2016_preproc1 = KagleSMS2016DatasetPreprocessors().preprocessor_1()

#название сценария будет использовано для названия каталога логов
test_scenarios = {
    'K2017_Email pr1 Tfidf1(ngram=(1,1))': #done 20/03
    ( kagle2017_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ), #( (corpus,y), (extractor_func, extractor_params), research_params )
    #'K2017_Email pr1 Tfidf1(ngram=(1,2))': 
    #( kagle2017_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,2)}), {} ), #done 20/03
    #'E_Email pr1 Tfidf1(ngram=(1,1))': 
    #( enron_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ),
    #'E_Email pr1 Tfidf1(ngram=(1,2))': 
    #( enron_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,2)}), {} ),
    #'K2016_SMS pr1 Tfidf1(ngram=(1,1))': 
    #( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ),
    #'K2016_SMS pr1 Tfidf1(ngram=(1,2))': 
    #( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,2)}), {} ), #done 20/03
    #'K2016_SMS pr1 Tfidf1(ngram=(1,3))': 
    #( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,3)}), {} ),
    #'K2016_SMS pr1 Tfidf1(ngram=(2,2))': 
    #( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(2,2)}), {} ),
    }

#в валидатор для тривиальных надо передавать всегда самый полный список алгоритмов, чтобы фильтрация результатов полноценно работала
algs_for_trivial_combis = {
        'ComplementNB_Default': ComplementNBAlg_Default(),
        'SGDClf_Default': SGDAlg_Default(),
        'SGDAlg_AdaptiveIters': SGDAlg_AdaptiveIters(),
        'SGDAlg_LogLoss': SGDAlg_LogLoss(),
        'ASGDAlg_Default': ASGDAlg_Default(),
        'NearestCentroid_Default': NearestCentroidAlg_Default(),
        'LinearSVC_Default': LinearSVCAlg_Default(),
        'LinearSVC_Balanced': LinearSVCAlg_Balanced(),
        'LinearSVCAlg_MoreSupports': LinearSVCAlg_MoreSupports(),
        'SVCAlg_RBF_Default': SVCAlg_RBF_Default(),
        'SVCAlg_RBF_Aggr': SVCAlg_RBF_Aggr(),
        'PAA_I_Default': PAA_I_Default(),
        'PAA_II_Default': PAA_II_Default(),
        'PAA_II_Balanced': PAA_II_Balanced(),
        'kNN_Default': KNeighborsAlg_Default(),
        'RandomForest_Default': RandomForestAlg_Default(),
        'RandomForest_Big': RandomForestAlg_Big(),
        'RandomForest_Medium': RandomForestAlg_Medium(),
        'RandomForest_Small': RandomForestAlg_Small(),
        'RandomForest_MDepth20': RandomForestAlg_MDepth20(),
        'RandomForest_MDepth30': RandomForestAlg_MDepth30(),
        'RandomForest_BigBootstrap75': RandomForestAlg_BigBootstrap75(),
        'RandomForest_Bootstrap90': RandomForestAlg_Bootstrap90(),
        'RandomForest_Balanced': RandomForestAlg_Balanced(),
        'Perceptron_Default': PerceptronAlg_Default()
        }

algs_without_ensembles = {
        'ComplementNB_Default': ComplementNBAlg_Default(),
        'SGDClf_Default': SGDAlg_Default(),
        'SGDAlg_LogLoss': SGDAlg_LogLoss(),
        'ASGDAlg_Default': ASGDAlg_Default(),
        'NearestCentroid_Default': NearestCentroidAlg_Default(),
        'LinearSVC_Default': LinearSVCAlg_Default(),
        'LinearSVC_Balanced': LinearSVCAlg_Balanced(),
        'LinearSVCAlg_MoreSupports': LinearSVCAlg_MoreSupports(),
        'SVCAlg_RBF_Default': SVCAlg_RBF_Default(),
        'SVCAlg_RBF_Aggr': SVCAlg_RBF_Aggr(),
        'PAA_I_Default': PAA_I_Default(),
        'PAA_II_Default': PAA_II_Default(),
        'PAA_II_Balanced': PAA_II_Balanced(),
        'kNN_Default': KNeighborsAlg_Default(),
        'Perceptron_Default': PerceptronAlg_Default()
        }

enabled_combinations_types = { #single algs (SA) validation включено по умолчанию
    'DC': True,
    'CC': True,
    'MC': True,
    'BAGC': True,
    'BOOSTC': False,
    'STACKC': False
    }

algs_dicts = {
    'TRIVIAL': algs_for_trivial_combis,
    'MC': algs_without_ensembles,
    'BAGC': algs_without_ensembles,
    'BOOSTC': None,
    'STACKC': None
    }

run_algs_validation()
#run_single_algs_test()