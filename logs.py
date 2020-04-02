# -*- coding: utf-8 -*-
import logging
from generic import *

import os
from pathlib import Path

class LogsFileProvider(object): #Singleton
    __instance = None #static
    LOGS_DIR = ServiceData.PROGRAM_DIR + r"\logs\.last\\"

    LOG_CONTENT_UNKN_HEADER = '//// UNKNOWN HEADER ////'

    class __LogsFileProvider:
        FORMATTER = logging.Formatter("%(message)s")
        def __init__(self):
            self.__init_loggers()

        def delete_old_log_file(self, log_path):
            try:
                os.remove(log_path)
            except OSError as exc:
                print(exc.filename + " " + exc.strerror)

        def __init_loggers(self): #для добавления логгера просто вписать в словарь,
            #названия логгеров должны отличаться (если 2 разных объекта ссылаются на логгеры с одним и тем же именем, то это на самом деле один и тот же логгер)
            
            self.loggers = {
                'ml_research_calculations': self.__init_logger('logger', LogsFileProvider.LOGS_DIR + "ml_research_calcs.log"),
                'ml_DC_sorted_f1': self.__init_logger('logger2', LogsFileProvider.LOGS_DIR + "ml_DC_sorted_f1.log"),
                'ml_DC_sorted_recall': self.__init_logger('logger3', LogsFileProvider.LOGS_DIR + "ml_DC_sorted_recall.log"),
                'ml_CC_sorted_f1': self.__init_logger('logger21', LogsFileProvider.LOGS_DIR + "ml_CC_sorted_f1.log"),
                'ml_CC_sorted_recall': self.__init_logger('logger31', LogsFileProvider.LOGS_DIR + "ml_CC_sorted_recall.log"),
                'ml_MAJ_sorted_f1': self.__init_logger('logger61', LogsFileProvider.LOGS_DIR + "ml_MAJ_sorted_f1.log"),
                'ml_MAJ_sorted_recall': self.__init_logger('logger71', LogsFileProvider.LOGS_DIR + "ml_MAJ_sorted_recall.log"),
                'ml_ALL_sorted_f1': self.__init_logger('logger40', LogsFileProvider.LOGS_DIR + "ml_ALL_sorted_f1.log"),
                'ml_ALL_sorted_recall': self.__init_logger('logger41', LogsFileProvider.LOGS_DIR + "ml_ALL_sorted_recall.log"),
                'ml_SA_sorted_f1': self.__init_logger('logger5', LogsFileProvider.LOGS_DIR + "ml_SA_sorted_f1.log")
                }

        def __init_logger(self,logger_name, logfile_path):
            Path(LogsFileProvider.LOGS_DIR).mkdir(parents=True, exist_ok=True)
            self.delete_old_log_file(logfile_path)
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.addHandler(self.__get_file_handler(logfile_path))
            logger.propagate = False
            return logger

        def __get_file_handler(self,file_path):
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(self.FORMATTER)
            return file_handler
                
        

    def __new__(cls):
        if cls.__instance == None:
            cls.__instance = LogsFileProvider.__LogsFileProvider()
        return cls.__instance

    @staticmethod
    def delete_log_files_on_hot(): #при работе программы файл занят ею и удалить его напрямую невозможно
        lfp = LogsFileProvider.__instance
        for logger_name in lfp.loggers:
            log_file_paths = []
            for handler in lfp.loggers[logger_name].handlers:
                log_file_paths.append(handler.baseFilename)
                handler.flush()
                handler.close()
                lfp.delete_old_log_file(handler.baseFilename)
            for log_file in log_file_paths:
                del lfp.loggers[logger_name].handlers[0]
                lfp.loggers[logger_name].addHandler(lfp.__get_file_handler(log_file))

    @staticmethod
    def close():
        LogsFileProvider.__instance = None

    @staticmethod
    def shutdown():
        logging.shutdown()
        LogsFileProvider.close()

    @staticmethod
    def log_named_info_block(logger, filtered_info_str, filter_params={}, log_header=LOG_CONTENT_UNKN_HEADER): 
        #param-ы фильтра в упакованном виде именно для вывода в лог
        logger.info(log_header) #print HEADER
        logger.info('filter_params: ' + str(filter_params)) #print params
        logger.info(filtered_info_str) #print info