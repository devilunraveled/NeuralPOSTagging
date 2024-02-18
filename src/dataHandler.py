"""
    This file contains all the necessary
    pre-processing required for the dataset
    before passing it to the model.
"""
from io import open
import conllu as Conllu

from config import Config
from base import Sentence

class DataHandler :
    def __init__(self, dataFileName : str) :
        self.dataFileName = dataFileName
        self.dataFilePath = Config.dataPath + self.dataFileName
        
        self.vocab = set(Config.specialTokens)

        self.data = {}

    def loadDataset(self, printSentences = False) :
        try :
            self.data = {}
            file = open(self.dataFilePath, "r", encoding="utf-8", errors='ignore')
            
            sentenceInformation = {}
            
            for sent in Conllu.parse_incr(file):
                sentenceInformation['POS'] = [token['upos'] for token in sent]
                self.vocab.update( [token['form'] for token in sent] )
                
                self.data[sent.metadata['sent_id']] = Sentence(sentenceInformation | sent.metadata )
                
                if printSentences :
                    print(self.data[sent.metadata['sent_id']])

            sentenceInformation.clear()
            
            file.close()
        except Exception as e :
            print(e)
            return -1


if __name__ == "__main__" :
    dataHandler = DataHandler("sample.conllu")
    dataHandler.loadDataset()
