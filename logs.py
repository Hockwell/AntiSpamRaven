# -*- coding: utf-8 -*-
import logging
import os

class LogsFileProvider(object): #Singleton
    instance = None #static
    class __LogsFileProvider:
        FORMATTER = logging.Formatter("%(message)s")

        def __init__(self):
            self.__init_loggers()

        def delete_old_log(self, log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass

        def __init_loggers(self): #для добавления логгера просто вписать сюда в виде поля
            #названия логгеров должны отличаться (параметр)
            self.ml_research_general = self.__init_log('logger', logfile_name="ml_research_general.log")
            self.ml_ODC_sorted_f1 = self.__init_log('logger2', logfile_name="ml_ODC_sorted_f1.log")
            self.ml_ODC_sorted_recall = self.__init_log('logger3', logfile_name="ml_ODC_sorted_recall.log")
            self.ml_OCC_sorted_f1 = self.__init_log('logger21', logfile_name="ml_OCC_sorted_f1.log")
            self.ml_OCC_sorted_recall = self.__init_log('logger31', logfile_name="ml_OCC_sorted_recall.log")
            self.ml_ODC_OCC_sorted_f1 = self.__init_log('logger40', logfile_name="ml_ODC_OCC_sorted_f1.log")
            self.ml_ODC_OCC_sorted_recall = self.__init_log('logger41', logfile_name="ml_ODC_OCC_sorted_recall.log")
            self.ml_single_algs_sorted_f1 = self.__init_log('logger5', logfile_name="ml_single_algs_sorted_f1.log")

        def __init_log(self,logger_name, logfile_name):
            self.delete_old_log(logfile_name)
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.addHandler(self.__get_file_handler(logfile_name))
            logger.propagate = False
            return logger
    
        def __get_file_handler(self,file_name):
            file_handler = logging.FileHandler(file_name)
            file_handler.setFormatter(self.FORMATTER)
            return file_handler

    def __new__(cls):
        if cls.instance is None:
            cls.instance = LogsFileProvider.__LogsFileProvider()
        return cls.instance

    