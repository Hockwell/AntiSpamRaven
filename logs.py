# -*- coding: utf-8 -*-
import logging

class LogsFileProvider(): #Singleton
    FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(message)s")
    instance = None #экземпляр текущего класса
    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(LogsFileProvider, cls).__new__(cls)
        return cls.instance
    
    def __init__(self):
        self.init_loggers()
        
    def init_loggers(self):
        self.logger_ml_processing = self.init_log('logger_main', logfile_name="ml_process.log")

    def init_log(self,logger_name, logfile_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.get_file_handler(logfile_name))
        logger.propagate = False
        return logger
    
    def get_file_handler(self,file_name):
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(self.FORMATTER)
        return file_handler

class LogsWriter():
	def write_dict(log_file, dictt):
		pass
	def write_list(log_file, listt):
		pass