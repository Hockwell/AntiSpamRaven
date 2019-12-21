# -*- coding: utf-8 -*-
import logging
import os

class LogsFileProvider(object): #Singleton
    instance = None #static
    class __LogsFileProvider:
        FORMATTER = logging.Formatter("%(message)s")

        def __init__(self):
            self.init_loggers()

        def delete_old_log(self, log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass

        def init_loggers(self): #для добавления логгера просто вписать сюда в виде поля
            #нужен, например, для записи информации по мере её поступления, чтобы в случае сбоя программы сохранились инфа для поиска ошибки
            #в процессах ML
            self.ml_research_general = self.init_log('logger', logfile_name="ml_research_general.log")
            self.ml_research_combis_sorted_f1 = self.init_log('logger2', logfile_name="ml_research_combis_f1.log")
            self.ml_research_combis_sorted_recall = self.init_log('logger3', logfile_name="ml_research_combis_recall.log")

        def init_log(self,logger_name, logfile_name):
            self.delete_old_log(logfile_name)
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.addHandler(self.get_file_handler(logfile_name))
            logger.propagate = False
            return logger
    
        def get_file_handler(self,file_name):
            file_handler = logging.FileHandler(file_name)
            file_handler.setFormatter(self.FORMATTER)
            return file_handler

    def __new__(cls):
        if cls.instance is None:
            cls.instance = LogsFileProvider.__LogsFileProvider()
        return cls.instance

    