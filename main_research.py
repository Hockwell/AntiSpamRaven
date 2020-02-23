#%%
import seaborn as sns
from sklearn.metrics import accuracy_score

import shutil
from os import listdir
from pathlib import Path

from datasets_preprocessing import *
from algs import *
from feature_extraction import *
from ml_algs_search import *
from generic import *

def set_libs_settings():
    ax = sns.set(style="darkgrid")

def run_single_algs_test():
    for alg in algs.items():
        y_pred = alg[1].learn_predict(X_train, X_test, y_train)
        print(alg[0],': ', accuracy_score(y_test, y_pred))
    print('//////////////////////////// single alg test done')

def run_tests_for_search_of_best_algs_combi():
    def run_searcher_on_dataset(X,y, k_folds, max_combination_length = 4):
        algs_searcher = AlgsBestCombinationSearcher()
        algs_searcher.run(X, y, k_folds, algs, max_combination_length)
        print('//////////////////////////// algs search done')
    
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

    def replace_results_to_specific_dir():
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
        run_searcher_on_dataset(X, y, **research_params, k_folds=10)
        replace_results_to_specific_dir() #они уже экспортированы, но лежат в общем каталоге по умолчанию, их нужно перенести

set_libs_settings()

#Работа с любым датасетом - это прохождение необходимых этапов: 1) предобработка, 2) извлечение признаков, 3) тестирование с комбинациями,
#поэтому и созданы "тестовые сценарии", любой сценарий здесь можно задать или легко модифицировать старые. 
#Сценарии можно отключать простым комментированием. Эти сценарии можно сделать совместимыми с любым исследовательским кодом.
#{ scenarios_name: ( (corpus,y), (extractor,{params}), {research_params} ) }
#препроцессинг вынесен в отдельный список по разным причинам, в том числе из-за необходимости делать тестовые сценарии с комбинациями датасетов
kagle2017_preproc1 = Kagle2017DatasetPreprocessors().preprocessor_1()
enron_preproc1 = EnronDatasetPreprocessors().preprocessor_1()
kagle2016_preproc1 = KagleSMS2016DatasetPreprocessors().preprocessor_1()

#название сценария будет использовано для названия каталога логов
test_scenarios = {
    #'K2017_Email pr1 Tfidf1(ngram=(1,1))': 
    #( kagle2017_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ),
    #'K2017_Email pr1 Tfidf1(ngram=(1,2))': 
    #( kagle2017_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {} ),
    #'E_Email pr1 Tfidf1(ngram=(1,2))': 
    #( enron_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,2)}), {} ),
    'K2016_SMS pr1 Tfidf1(ngram=(1,1))': 
    ( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,1)}), {'max_combination_length': 5} ),
    #'K2016_SMS pr1 Tfidf1(ngram=(1,3))': 
    #( kagle2016_preproc1, (FeatureExtractors.extractor_tfidf_1, {'ngram_range':(1,3)}), {} ),
    }

algs = {
        'ComplementNB_Default': ComplementNBAlg_Default(),
        'SGDClf_Default': SGDAlg_Default(),
        'SGDAlg_AdaptiveIters': SGDAlg_AdaptiveIters(),
        'SGDAlg_LogLoss': SGDAlg_LogLoss(),
        'ASGDAlg_Default': ASGDAlg_Default(),
        #'NearestCentroid_Default': NearestCentroidAlg_Default(),
        #'LinearSVC_Default': LinearSVCAlg_Default(),
        #'LinearSVCAlg_Balanced': LinearSVCAlg_Balanced(),
        #'SVCAlg_RBF_Default': SVCAlg_RBF_Default(),
        #'SVCAlg_RBF_Aggr': SVCAlg_RBF_Aggr(),
        #'PassiveAggressiveClf_Default': PassiveAggressiveAlg_Default(),
        #'RidgeClf_Default': RidgeAlg_Default(),
        #'KNeighborsClf_Default': KNeighborsAlg_Default(),
        #'RandomForest_Default': RandomForestAlg_Default(),
        #'RandomForest_Mod1': RandomForestAlg_Mod1(),
        #'RandomForest_Mod2': RandomForestAlg_Mod2(),
        #'RandomForest_Mod3': RandomForestAlg_Mod3(),
        #'RandomForest_Mod4': RandomForestAlg_Mod4(),
        #'Perceptron_Default': PerceptronAlg_Default()
        }

run_tests_for_search_of_best_algs_combi()
#run_single_algs_test()


#this function computes subset accuracy
#accuracy_score(y_test, y_pred)
#accuracy_score(y_test, y_pred, normalize=False)

#Making the Confusion Matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

# Applying k-Fold Cross Validation
#accuracies = cross_val_score(estimator = alg.clf, X = X_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()


