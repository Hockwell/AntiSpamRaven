# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os
from abc import ABC, abstractmethod

#можно использовать родительский класс, а можно писать полностью свой препроцессор, в том числе без сохранения данных препроцессинга
class DatasetsPreprocessors(ABC):
    def __init__(self):
        self._PREPROC_RESULTS_PATH = r"C:\Users\volnu\OneDrive\Data\Dev\Src\Med\AntiSpamRaven\preproc_results\\"
        self._DATASETS_PATH = r"C:\Users\volnu\OneDrive\Data\Dev\Src\Med\AntiSpamRaven\datasets\\"
        self._DATA_FILE_EXTENSION = ".csv"

        #load_saved_preproc_data() не помещён сюда, ибо архитектурно мы могли сохранить больше данных после препроцессинга, это зависит 
        #от датасета и извлекаемых далее фичей
    def _preprocess(self, load_saved_preproc_data_func, run_preprocessing_func): #осуществляет управлением сохранением результатов препроцессинга
        try:
            dataset_corpus, y = load_saved_preproc_data_func()
        except IOError:
            try:
                os.remove(self._PREPROC_CORPUS_FILE_PATH)
                os.remove(self._PREPROC_Y_FILE_PATH)
            except OSError:
                pass
            dataset_corpus, y = run_preprocessing_func()
        return dataset_corpus, y

class Kagle2017DatasetPreprocessors(DatasetsPreprocessors): #эти классы должны быть Singleton-ами, но сделать наследование от класса DP
   #при этом будет невозможно. Реализация в виде статических классов более громоздкая. 
    def __init__(self):
        super().__init__()
        self._DATASET_NAME = "emails_kagle_2017"
        self._DATASET_FILE_NAME = self._DATASET_NAME + self._DATA_FILE_EXTENSION
        self._DATASET_PATH = self._DATASETS_PATH + self._DATASET_FILE_NAME

    def preprocessor_1(self):
        self._PREPROC_FILES_SUFFIX = "_preproc1"
        self._PREPROC_CORPUS_FILE_NAME = self._DATASET_NAME + self._PREPROC_FILES_SUFFIX + "_corpus" + self._DATA_FILE_EXTENSION;
        self._PREPROC_Y_FILE_NAME = self._DATASET_NAME + self._PREPROC_FILES_SUFFIX + "_y" + self._DATA_FILE_EXTENSION;
        self._PREPROC_CORPUS_FILE_PATH = self._PREPROC_RESULTS_PATH + self._PREPROC_CORPUS_FILE_NAME
        self._PREPROC_Y_FILE_PATH = self._PREPROC_RESULTS_PATH + self._PREPROC_Y_FILE_NAME

        def load_saved_preproc_data():
            dataset_corpus = pd.read_csv(filepath_or_buffer = self._PREPROC_CORPUS_FILE_PATH, names = ['text'])['text']
            y = pd.read_csv(filepath_or_buffer = self._PREPROC_Y_FILE_PATH, names = ['y'])['y']
            return dataset_corpus,y

        def run_preprocessing():
            def save_preproc_data():
                dataset_corpus.to_csv(path_or_buf = self._PREPROC_CORPUS_FILE_PATH, index = False, columns = ['text'])
                y.to_csv(path_or_buf = self._PREPROC_Y_FILE_PATH, index = False)

            nltk.download('stopwords')
            raw_data = pd.read_csv(self._DATASET_PATH)
            #Checking for duplicates and removing them
            dataset = raw_data.drop_duplicates()
            #Checking for any null entries in the dataset
            #print (pd.DataFrame(dataset.isnull().sum()))

            #Using Natural Language Processing to cleaning the text to make one corpus
            # Cleaning the texts
            #Every mail starts with 'Subject :' will remove this from each text 
            dataset['text'] = dataset['text'].map(lambda text: text[8:])
            #всё, что не является англ. буквами и цифрами, заменяем на пробелы или не меняем, если под шаблон не попадает
            #далее делаем весь текст строчного регистра и делим его на слова по пробелам - получаем список разделенных пробелами слов/цифр
            #теперь dataset - это слова
            dataset['text'] = dataset['text'].map(lambda text: re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
            #используя алгоритм стемминга Портера
            ps = PorterStemmer()
            #Удаляем из списка слов каждого семпла стоп-слова (используя ntlk). Технически мы из старых семплов 
            #делаем новые, с отфильтрованным контентом, далее пропускаем через стемминг с помощью map(), имеем список слов,
            #потом соединяем эл-ты списка пробелами - получаем на один семпл одну строку 
            #со словами вместо списка слов: .join(map(lambda_a,list from lambda_b(sample_words)))
            dataset_corpus = dataset['text'].apply(lambda sample_words:' '.join( list(map( lambda word: ps.stem(word), 
                            list(filter(lambda text: text not in set(stopwords.words('english')), sample_words)) )) ))
            y = dataset.iloc[:, 1]

            save_preproc_data()
            #получаем корпус слов для каждого семпла - каждый семпл выражен списком необходимых слов (а не всех тех, что содержались в нём)
            return dataset_corpus,y
        
        return self._preprocess(load_saved_preproc_data, run_preprocessing)

class EnronDatasetPreprocessors(DatasetsPreprocessors):
    def __init__(self):
        super().__init__()
        self._DATASET_NAME = "emails_enron_99-05"
        self._DATASET_FILE_NAME = self._DATASET_NAME + self._DATA_FILE_EXTENSION
        self._DATASET_PATH = self._DATASETS_PATH + self._DATASET_FILE_NAME

    def preprocessor_1(self):
        self._PREPROC_FILES_SUFFIX = "_preproc1"
        self._PREPROC_CORPUS_FILE_NAME = self._DATASET_NAME + self._PREPROC_FILES_SUFFIX + "_corpus" + self._DATA_FILE_EXTENSION;
        self._PREPROC_Y_FILE_NAME = self._DATASET_NAME + self._PREPROC_FILES_SUFFIX + "_y" + self._DATA_FILE_EXTENSION;
        self._PREPROC_CORPUS_FILE_PATH = self._PREPROC_RESULTS_PATH + self._PREPROC_CORPUS_FILE_NAME
        self._PREPROC_Y_FILE_PATH = self._PREPROC_RESULTS_PATH + self._PREPROC_Y_FILE_NAME
        
        def load_saved_preproc_data():
            dataset_corpus = pd.read_csv(filepath_or_buffer = self._PREPROC_CORPUS_FILE_PATH, names = ['text'])['text']
            y = pd.read_csv(filepath_or_buffer = self._PREPROC_Y_FILE_PATH, names = ['y'])['y']
            return dataset_corpus,y

        def run_preprocessing():
            def mark_useless_samples(): #as np.nan
                #dataset['text'].replace('', np.nan, inplace=True)
                mask = dataset['text'].str.len() > 3
                return dataset[mask]

            def save_preproc_data():
                dataset['text'].to_csv(path_or_buf = self._PREPROC_CORPUS_FILE_PATH, index = False, columns = ['text'])
                y.to_csv(path_or_buf = self._PREPROC_Y_FILE_PATH, index = False)
            
            nltk.download('stopwords')
            raw_data = pd.read_csv(self._DATASET_PATH)
            raw_data = raw_data.drop('Unnamed: 0', axis=1)
            raw_data = raw_data.drop('label', axis=1)
            #print(raw_data.head())
            dataset = raw_data.drop_duplicates()
            dataset['text'] = dataset['text'].map(lambda sample: sample[8:]) #delete "Subject:"
            dataset['text'] = dataset['text'].map(lambda sample_words: re.sub('[^a-zA-Z0-9]+', ' ',sample_words)).apply(lambda x: (x.lower()).split())
            ps = PorterStemmer()
            dataset['text'] = dataset['text'].apply(lambda sample_words:' '.join( list(map(lambda word: ps.stem(word), 
                            list(filter(lambda text: text not in set(stopwords.words('english')), sample_words)))) ))
            #удалим семплы длинной <3 из датасета
            dataset = mark_useless_samples()
            dataset = dataset.dropna(axis=0)
            y = dataset.iloc[:, 1]
            save_preproc_data()
            return dataset['text'],y
        
        return self._preprocess(load_saved_preproc_data, run_preprocessing)
