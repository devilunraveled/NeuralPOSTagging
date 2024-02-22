"""
    This file contains all the necessary
    pre-processing required for the dataset
    before passing it to the model.
"""
from io import open
import conllu as Conllu

from src.config import Config
from src.base import Sentence

class DataHandler :
    def __init__(self, dataFileName : str) :
        self.dataFileName = dataFileName
        self.dataFilePath = Config.dataPath + self.dataFileName

        self.trainDataFilePath = self.dataFilePath + Config.trainTag + Config.fileFormat
        self.testDataFilePath = self.dataFilePath + Config.testTag + Config.fileFormat
        self.validationDataFilePath = self.dataFilePath + Config.validationTag + Config.fileFormat

        self.vocab = set(Config.specialTokens)

        self.trainData = {}
        self.testData = {}
        self.validationData = {}

    def loadTrainDataset(self, printSentences = False) :
        try :
            file = open(self.trainDataFilePath, "r", encoding="utf-8", errors='ignore')
            
            sentenceInformation = {}
            
            for sent in Conllu.parse_incr(file):
                sentenceInformation['POS'] = [token['upos'] for token in sent]
                self.vocab.update( [token['form'] for token in sent] )
                
                self.trainData[sent.metadata['sent_id']] = Sentence(sentenceInformation | sent.metadata )
                
                if printSentences :
                    print(self.trainData[sent.metadata['sent_id']])

            sentenceInformation.clear()
            
            file.close()
        except Exception as e :
            print(e)
            return -1
    
    def loadTestDataSet(self):
        try :
            self.testData = {}
            file = open(self.testDataFilePath, "r", encoding="utf-8", errors='ignore')
            
            sentenceInformation = {}
            
            for sent in Conllu.parse_incr(file):
                sentenceInformation['POS'] = [token['upos'] for token in sent]

                self.testData[sent.metadata['sent_id']] = Sentence(sentenceInformation | sent.metadata )
                
            sentenceInformation.clear()
            
            file.close()
        except Exception as e :
            print(e)
            return -1
    
    def loadValidationDataSet(self):
        try :
            self.validationData = {}
            file = open(self.validationDataFilePath, "r", encoding="utf-8", errors='ignore')
            
            sentenceInformation = {}
            
            for sent in Conllu.parse_incr(file):
                sentenceInformation['POS'] = [token['upos'] for token in sent]
                # self.vocab.update( [token['form'] for token in sent] )
                
                self.validationData[sent.metadata['sent_id']] = Sentence(sentenceInformation | sent.metadata )
                
            sentenceInformation.clear()
            
            file.close()
        except Exception as e :
            print(e)
            return -1
    
    def loadDataset(self, loadTrain : bool = True, loadTest : bool = True, loadValidation : bool = True) :
        try :
            if loadTrain :
                self.loadTrainDataset()
            if loadTest :
                self.loadTestDataSet()
            if loadValidation :
                self.loadValidationDataSet()
        except Exception as e :
            print(e)
            return -1

    def createLookUpTable(self):
        try:
            self.lookUpTable = {}

            for index, token in enumerate(self.vocab):
                self.lookUpTable[token] = index

        except Exception as E:
            print(E)
            return -1

if __name__ == "__main__" :
    dataHandler = DataHandler("sample")
    dataHandler.loadDataset(loadTrain = True, loadTest = False, loadValidation = False)
