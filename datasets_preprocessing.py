# -*- coding: utf-8 -*-
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os

PREPROC_FILES_DIR = r"C:\Users\volnu\OneDrive\Data\Dev\Src\Med\AntiSpamRaven\preproc_results\\"

class Kagle2017DatasetPreprocessors(object):
    #не сделан статическим, чтобы можно было создавать разные экземпляры с разными параметрами предобработки
    #результаты препроцессинга сохраняются на диске, если их нет - проводим его заново
    PATH = r"C:\Users\volnu\OneDrive\Data\Dev\Src\Med\AntiSpamRaven\datasets\emails_kagle_2017.csv"
    PREPROC_CORPUS_FILE_PATH = PREPROC_FILES_DIR + "emails_kagle_2017_corpus.csv"
    PREPROC_Y_FILE_PATH = PREPROC_FILES_DIR + "emails_kagle_2017_y.csv"
    
    def preprocessor_1(self):
        dataset_corpus = None
        raw_data = pd.read_csv(self.PATH)
        try:
            #print('corpus and y loading began')
            dataset_corpus = pd.read_csv(filepath_or_buffer = self.PREPROC_CORPUS_FILE_PATH, names = ['text'])['text']
            y = pd.read_csv(filepath_or_buffer = self.PREPROC_Y_FILE_PATH, names = ['y'])['y']
            #print('corpus and y loading done')
        except IOError:
            try:
                os.remove(self.PREPROC_CORPUS_FILE_PATH)
                os.remove(self.PREPROC_Y_FILE_PATH)
            except OSError:
                pass
            nltk.download('stopwords')
            #Checking for duplicates and removing them
            dataset = raw_data.drop_duplicates()
            #Checking for any null entries in the dataset
            #print (pd.DataFrame(dataset.isnull().sum()))

            #Using Natural Language Processing to cleaning the text to make one corpus
            # Cleaning the texts
            #Every mail starts with 'Subject :' will remove this from each text 
            dataset['text'] = dataset['text'].map(lambda text: text[1:])
            #всё, что не является англ. буквами и цифрами, заменяем на пробелы или не меняем, если под шаблон не попадает
            #далее делаем весь текст строчного регистра и делим его на слова по пробелам - получаем список разделенных пробелами слов/цифр
            #теперь dataset - это слова
            dataset['text'] = dataset['text'].map(lambda text: re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
            #используя алгоритм стемминга Портера
            ps = PorterStemmer()
            #Удаляем из списка слов каждого семпла стоп-слова (используя ntlk). Технически мы из старых семплов 
            #делаем новые, с отфильтрованным контентом, далее пропускаем через стемминг с помощью map(),
            #потом соединяем эл-ты списка пробелами - получаем на один семпл одну строку 
            #со словами вместо списка слов: .join(map(lambda_a,list from lambda_b(sample_words)))
            dataset_corpus = dataset['text'].apply(lambda sample_words:' '.join( list(map(lambda word: ps.stem(word), 
                            list(filter(lambda text: text not in set(stopwords.words('english')), sample_words)))) ))
            y = dataset.iloc[:, 1]
            
            dataset_corpus.to_csv(path_or_buf = self.PREPROC_CORPUS_FILE_PATH, index = False, columns = ['text'])
            y.to_csv(path_or_buf = self.PREPROC_Y_FILE_PATH, index = False)
        print('//////////////////////////// preprocessing done')
        #получаем корпус слов для каждого семпла - каждый семпл выражен списком необходимых слов (а не всех тех, что содержались в нём)
        return dataset_corpus,y

#class EnronDatasetsPreprocessors(object): #подойдёт для всех Энроновских датасетов, сделай препроцессинг по параметру, который 
#    #является номером датасета, который нужно препроцессить
    
#    PATH = r"C:\Users\volnu\OneDrive\Data\Dev\Src\Med\AntiSpamRaven\datasets\"
#    PREPROC_CORPUS_FILE_PATH = PREPROC_FILES_DIR + "emails_kagle_2017_corpus.csv"
#    PREPROC_Y_FILE_PATH = PREPROC_FILES_DIR + "emails_kagle_2017_y.csv"
    
#    def preprocessor_1(self):
        
#        print('//////////////////////////// preprocessing done')
#        #получаем корпус слов для каждого семпла - каждый семпл выражен списком необходимых слов (а не всех тех, что содержались в нём)
#        return dataset_corpus,y