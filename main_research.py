#%%
import seaborn as sns

from sklearn.metrics import accuracy_score


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

def run_algs_best_combination_searcher(algs,X,y, k_folds):
    algs_searcher = AlgsBestCombinationSearcher()
    algs_searcher.run(X, y, k_folds, algs)
    print('//////////////////////////// algs search done')
       
def visualize_dataset(y):
    sns.countplot(y=y)

set_libs_settings()
#dataset_corpus, y = Kagle2017DatasetPreprocessors().preprocessor_1()
dataset_corpus, y = EnronDatasetPreprocessors().preprocessor_1()
print('//////////////////////////// preprocessing done')
X = FeatureExtractors.extractor_tfidf_1(dataset_corpus) #corpus -> X
print('//////////////////////////// feature extraction done')
X_train, y_train, X_test, y_test = DatasetInstruments.make_shuffle_stratified_split_on_folds(X,y, test_size = 0.25, n_splits=1)[0]

#visualize_dataset(y)
#visualize_dataset(y_train)
#visualize_dataset(y_test)

#search of best algs combination
algs = {
        'ComplementNB_Default': ComplementNBAlg_Default(),
        'SGDClf_Default': SGDAlg_Default(),
        'NearestCentroid_Default': NearestCentroidAlg_Default(),
        #'LinearSVC_Default': LinearSVCAlg_Default(),
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
run_single_algs_test()
#run_algs_best_combination_searcher(algs, X, y, k_folds=10)

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


