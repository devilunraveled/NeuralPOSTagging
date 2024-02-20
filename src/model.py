"""
    This is the general class that will serve as the base 
    class for all the models in this assignment.
"""
import pickle
from config import Config

class Model :
    def __init__(self, modelName = None) :
        self.modelName = "Model" if modelName is None else modelName
        self.modelFileName = Config.modelSavePath + self.modelName

    def test(self) :
        pass
    
    def saveModel(self):
        try :
            pickle.dump(self, open(self.modelFileName + Config.fileFormat, "wb"))
            return 1
        except :
            return -1
    
    def loadModel(self):
        try :
            loaded_model = pickle.load(open(self.modelFileName + Config.fileFormat, "rb"))
            self.__dict__.update(loaded_model.__dict__)
            print(f"Model Loaded from {self.modelFileName} Successfully!")
            return 1
        except :
            return -1

    # Evaluation Related Functionality
    def getEvaluationResults(self) :
        pass
