# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

class Kagle2017DatasetPreprocessors(object):
    #не сделан статическим, чтобы можно было создавать разные экземпляры с разными параметрами предобработки
    PATH = r"C:\Users\volnu\OneDrive\Data\Dev\Src\Med\AntiSpamRaven\datasets\emails_kagle_2017.csv"
    def preprocessor_1(self):
        raw_data = pd.read_csv(self.PATH)
        nltk.download('stopwords')
        
        #Checking for duplicates and removing them
        dataset = raw_data.drop_duplicates()
        dataset.shape  #(5695, 2)
        #Checking for any null entries in the dataset
        #print (pd.DataFrame(dataset.isnull().sum()))
        '''
        text  0
        spam  0
        '''
        #Using Natural Language Processing to cleaning the text to make one corpus
        # Cleaning the texts
        #Every mail starts with 'Subject :' will remove this from each text 
        dataset['text'] = dataset['text'].apply(lambda text: text[1:])
        #всё, что не является англ. буквами и цифрами, заменяем на пробелы или не меняем, если под шаблон не попадает
        #далее делаем весь текст строчного регистра и делим его на слова по пробелам - получаем список разделенных пробелами слов/цифр
        #теперь dataset - это слова
        dataset['text'] = dataset['text'].apply(lambda text: re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
        #используя алгоритм стемминга Портера, создаем словарь - корпус слов 
        ps = PorterStemmer()
        #Удаляем из списка слов каждого семпла стоп-слова (используя ntlk). 
        corpus = dataset['text'].apply(lambda sample_words:' '.join( list(map(lambda word: ps.stem(word), list(filter(lambda text: text not in set(stopwords.words('english')), sample_words)))) ))
        
        return corpus