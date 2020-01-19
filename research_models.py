from abc import ABC, abstractmethod


class ResearchModel(ABC): #RM
    @abstractmethod
    def run_preprocessing(self, input):
        pass

    @abstractmethod
    def extract_features(self, input):
        pass

    @abstractmethod
    def learn(self, input):
        pass

    @abstractmethod
    def predict(self, input):
        pass

class AnySpam_RM(ResearchModel): #обнаруживает любой спам, данная модель создана для доказательства, что лучше для обнаружения
    #спама разных возрастов использовать больше более узконаправленных моделей, чем более обобщённые по диапазону типов спама
    pass

class NewSpam_RM(ResearchModel):
    pass

class OldSpam_RM(ResearchModel):
    pass