"""
    This is the general class that will serve as the base 
    class for all the models in this assignment.
"""
import pickle
from src.config import Config

class Model :
    def __init__(self, modelName = None) :
        self.savePlots = False
        self.modelName = "Model" if modelName is None else modelName
        self.modelFileName = self.modelName
        self.modelFileName += f"_EPOCHS={Config.epochs}"
        self.modelFileName += f"_BATCH_SIZE={Config.miniBatchSize}"
    
    def saveModel(self):
        try :
            pickle.dump(self, open(Config.modelSavePath + self.modelFileName + Config.modelFileExtension, "wb"))
            print(f"Model Saved at {self.modelFileName}.")
            return 1
        except Exception as e:
            raise(e)
            return -1
    
    def loadModel(self):
        try :
            loaded_model = pickle.load(open(Config.modelSavePath + self.modelFileName + Config.modelFileExtension, "rb"))
            self.__dict__.update(loaded_model.__dict__)
            print(f"Model Loaded from {self.modelFileName}.")
            return 1
        except :
            return -1
    
    def trainModel(self, numEpochs = Config.epochs):
        try :
            return 1
        except :
            return None
    def setModelName(self, name : str):
        self.modelName = name

    def setParams(self, **parameters):
        try :
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self
        except Exception as e:
            print(e)
            return self

    def completeEvaluation(self) :
        return {}
