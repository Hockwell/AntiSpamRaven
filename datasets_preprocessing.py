# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os
from abc import ABC, abstractmethod

from generic import *

#!Чтобы задействовать новый препроцессор взамен старого для конкретного препроцессора и его датасета, необходимо удалить сохранённые данные препроцессинга вручную
#по пути _PREPROC_RESULTS_PATH. Программа сохраняет последний результат препроцессинга для каждого препроцессора датасета в отдельности.

#можно использовать родительский класс, а можно писать полностью свой препроцессор, в том числе без сохранения данных препроцессинга
class DatasetPreprocessors(ABC):
    def __init__(self, dataset_name, preproc_files_suffix):
        pass
    def __init__(self, dataset_name, preproc_files_suffix):
        self._PREPROC_RESULTS_PATH = ServiceData.PROGRAM_DIR + r"\preproc_results\\"
        self._DATASETS_PATH = ServiceData.PROGRAM_DIR + r"\datasets\\"
        self._DATA_FILE_EXTENSION = ".csv"
        self._DATASET_NAME = dataset_name
        self._DATASET_FILE_NAME = self._DATASET_NAME + self._DATA_FILE_EXTENSION
        self._DATASET_PATH = self._DATASETS_PATH + self._DATASET_FILE_NAME
        self._PREPROC_FILES_SUFFIX = preproc_files_suffix
        self._PREPROC_DATASET_FILE_NAME = self._DATASET_NAME + self._PREPROC_FILES_SUFFIX + "_bow" + self._DATA_FILE_EXTENSION;
        self._PREPROC_Y_FILE_NAME = self._DATASET_NAME + self._PREPROC_FILES_SUFFIX + "_y" + self._DATA_FILE_EXTENSION;
        self._PREPROC_DATASET_FILE_PATH = self._PREPROC_RESULTS_PATH + self._PREPROC_DATASET_FILE_NAME
        self._PREPROC_Y_FILE_PATH = self._PREPROC_RESULTS_PATH + self._PREPROC_Y_FILE_NAME

    @staticmethod
    def crop_samples_from_left(trimmed_text, dataset, col_with_text):
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample: sample[len(trimmed_text):])
        return dataset

    @staticmethod
    def _run_general_preprocessor_1(raw_data, col_with_text, min_sample_length = 2): #работает с датасетами, где есть лишь столбцы label и text
        def detect_useless_samples(dataset): #as np.nan(#)
            dataset.loc[:,(col_with_text)].replace("", np.nan, inplace=True)
            mask = dataset[col_with_text].str.len() >= min_sample_length
            dataset = dataset[mask].dropna(axis=0)
            return dataset

        nltk.download('stopwords')
        dataset = raw_data.drop_duplicates()
        #Replace email addresses with 'emailaddr'
        #Replace URLs with 'httpaddr'
        #Replace money symbols with 'moneysymb'
        #Replace phone numbers with 'phonenumbr'
        #Replace numbers with 'numbr'
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample_words: re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr' ,sample_words))
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample_words: re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr' ,sample_words))
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample_words: re.sub('£|\$', 'moneysymb' ,sample_words))
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample_words: re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', sample_words))
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample_words: re.sub('\d+(\.\d+)?', 'numbr' ,sample_words))
        #Remove all punctuations
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample_words: re.sub('[^\w\d\s]', ' ' ,sample_words))
        #всё, что не является англ. буквами и цифрами, заменяем на пробелы или не меняем, если под шаблон не попадает
            #далее делаем весь текст строчного регистра и делим его на слова по пробелам - получаем список разделенных пробелами слов/цифр
            #теперь dataset - это семплы из списков слов
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].map(lambda sample_words: re.sub('[^a-zA-Z0-9]+', ' ',sample_words)).apply(lambda x: (x.lower()).split())
        #используя алгоритм стемминга Портера
        ps = PorterStemmer()
        #Удаляем из списка слов каждого семпла стоп-слова (используя ntlk). Технически мы из старых семплов 
            #делаем новые, с отфильтрованным контентом, далее пропускаем через стемминг с помощью map(), имеем список слов,
            #потом соединяем эл-ты списка пробелами - получаем на один семпл одну строку 
            #со словами вместо списка слов: .join(map(lambda_a,list from lambda_b(sample_words)))
        dataset.loc[:,(col_with_text)] = dataset.loc[:,(col_with_text)].apply(lambda sample_words:' '.join( list(map(lambda word: ps.stem(word), 
                        list(filter(lambda text: text not in set(stopwords.words('english')), sample_words)))) ))
        dataset = detect_useless_samples(dataset)
        return dataset

    def _preprocess(self, load_saved_preproc_data_func, run_preprocessing_func): #осуществляет управлением сохранением результатов препроцессинга
        try:
            dataset_bow, y = load_saved_preproc_data_func()
        except IOError:
            try:
                os.remove(self._PREPROC_DATASET_FILE_PATH)
                os.remove(self._PREPROC_Y_FILE_PATH)
            except OSError:
                pass
            dataset_bow, y = run_preprocessing_func()
        return dataset_bow, y

class Kagle2017DatasetPreprocessors(DatasetPreprocessors): #эти классы должны быть Singleton-ами, но сделать наследование от класса DP
   #при этом будет невозможно. Реализация в виде статических классов более громоздкая. 
    def __init__(self):
        pass     

    def preprocessor_1(self):
        def load_saved_preproc_data():
            dataset_bow = pd.read_csv(filepath_or_buffer = self._PREPROC_DATASET_FILE_PATH, names = ['text'])['text']
            y = pd.read_csv(filepath_or_buffer = self._PREPROC_Y_FILE_PATH, names = ['y'])['y']
            return dataset_bow,y

        def run_preprocessing():
            def save_preproc_data():
                dataset.loc[:,('text')].to_csv(path_or_buf = self._PREPROC_DATASET_FILE_PATH, index = False, columns = ['text'])
                y.to_csv(path_or_buf = self._PREPROC_Y_FILE_PATH, index = False)

            raw_data = pd.read_csv(self._DATASET_PATH)

            dataset = DatasetPreprocessors._run_general_preprocessor_1(raw_data, 'text', 10)
            dataset = DatasetPreprocessors.crop_samples_from_left('subject', dataset, 'text')
            y = dataset.iloc[:, 1]

            save_preproc_data()
            #получаем корпус слов для каждого семпла - каждый семпл выражен списком необходимых слов (а не всех тех, что содержались в нём)
            return dataset.loc[:,('text')],y
        super().__init__("emails_kagle_2017" , "_preproc1")
        return self._preprocess(load_saved_preproc_data, run_preprocessing)

class EnronDatasetPreprocessors(DatasetPreprocessors):
    def __init__(self):
        pass

    def preprocessor_1(self):
        def load_saved_preproc_data():
            dataset_bow = pd.read_csv(filepath_or_buffer = self._PREPROC_DATASET_FILE_PATH, names = ['text'])['text']
            y = pd.read_csv(filepath_or_buffer = self._PREPROC_Y_FILE_PATH, names = ['y'])['y']
            return dataset_bow,y

        def run_preprocessing():
            def save_preproc_data():
                dataset.loc[:,('text')].to_csv(path_or_buf = self._PREPROC_DATASET_FILE_PATH, index = False, columns = ['text'])
                y.to_csv(path_or_buf = self._PREPROC_Y_FILE_PATH, index = False)
            
            raw_data = pd.read_csv(self._DATASET_PATH)
            raw_data = raw_data.drop('Unnamed: 0', axis=1)
            raw_data = raw_data.drop('label', axis=1)
            dataset = DatasetPreprocessors._run_general_preprocessor_1(raw_data, 'text', 10)
            dataset = DatasetPreprocessors.crop_samples_from_left('subject', dataset, 'text')
            y = dataset.iloc[:, 1]
            save_preproc_data()
            return dataset.loc[:,('text')],y
        super().__init__("emails_enron_99-05" , "_preproc1")
        return self._preprocess(load_saved_preproc_data, run_preprocessing)

class KagleSMS2016DatasetPreprocessors(DatasetPreprocessors):
    def __init__(self):
        pass

    def preprocessor_1(self):     
        def load_saved_preproc_data():
            dataset_bow = pd.read_csv(filepath_or_buffer = self._PREPROC_DATASET_FILE_PATH, names = ['v2'])['v2']
            y = pd.read_csv(filepath_or_buffer = self._PREPROC_Y_FILE_PATH, names = ['v1'])['v1']
            return dataset_bow,y

        def run_preprocessing():
            def save_preproc_data():
                dataset.loc[:,('v2')].to_csv(path_or_buf = self._PREPROC_DATASET_FILE_PATH, index = False, columns = ['v2'])
                y.to_csv(path_or_buf = self._PREPROC_Y_FILE_PATH, index = False)
            
            nltk.download('stopwords')
            raw_data = pd.read_csv(self._DATASET_PATH, encoding='latin-1')
            raw_data = raw_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
            dataset = DatasetPreprocessors._run_general_preprocessor_1(raw_data, 'v2')
            dataset = dataset.replace(['ham','spam'],[0, 1])
            y = dataset.iloc[:, 0]
            save_preproc_data()
            return dataset.loc[:,('v2')],y
        super().__init__("sms_kagle_2016" , "_preproc1")
        return self._preprocess(load_saved_preproc_data, run_preprocessing)