from model import Model
from dataHandler import DataHandler
from base import Sentence, DataPoint
from src.config import Config


class FeedForwardNeuralNetworkDataPoint(DataPoint) :
    def __init__(self, dataPoint, lookUp : dict, position, vocabSize) :
        self.dataPoint = [token[0] for token in dataPoint], dataPoint[position][1]
        self.getTokenIndex = lookUp
        self.position = position
        self.vocabSize = vocabSize

    def getEmbedding(self):
        try :
            dataPointEmbedding = []

            for token in self.dataPoint[0]:
                tokenIndex = self.getTokenIndex[token]
                dataPointEmbedding.extend( [1 if i == tokenIndex else 0 for i in range(self.vocabSize)] )
            
            # print(f"{self.dataPoint=} \n {dataPointEmbedding=}\n")
            return dataPointEmbedding
        except Exception as e:
            print(e)
            return []


class FeedForwardNeuralNetworkDataHandler(DataHandler) :
    def __init__(self, dataFileName : str, previousContextSize : int = Config.previousContextSize, futureContextSize : int = Config.futureContextSize) :
        super().__init__(dataFileName=dataFileName)
        self.previousContextSize = previousContextSize
        self.futureContextSize = futureContextSize

    def createRawDataPointsFromSentence(self, sentence):
        try :
            assert isinstance(sentence, Sentence)
            sentenceTags = sentence.getTokensWithLabels()

            startPadding = [('<start>', None)] * self.previousContextSize
            endPadding = [('<end>', None)] * self.futureContextSize
            
            sentenceTags = startPadding + list(sentenceTags) + endPadding
            
            sentenceBegining = self.previousContextSize
            sentenceEnding = len(sentenceTags) - self.futureContextSize
            
            dataPoints = []

            for i in range(sentenceBegining, sentenceEnding):
                dataPoints.append( sentenceTags[i - self.previousContextSize : i + self.futureContextSize + 1] )
            
            return dataPoints

        except Exception as e:
            print(e)
            return []

    def createRawDataPoints(self):
        try :
            self.rawDataPoints = []
            self.loadDataset()
            
            for sentence in self.data.values():
                self.rawDataPoints.extend(self.createRawDataPointsFromSentence(sentence))
            
            return self.rawDataPoints
        except Exception as e:
            print(e)
            return []
    
    def getDataPoints(self):
        try :
            self.createRawDataPoints()
            self.createLookUpTable()
            
            self.dataPoint = []

            for rawDataPoint in self.rawDataPoints:
                newDataPoint = FeedForwardNeuralNetworkDataPoint( dataPoint=rawDataPoint, lookUp=self.lookUpTable, position=self.previousContextSize, vocabSize=len(self.vocab))
                self.dataPoint.append(newDataPoint.getEmbedding())
            print(self.dataPoint)

        except Exception as e:
            print(e)
            return []
    
    def createLookUpTable(self):
        try:
            self.lookUpTable = {}

            for index, token in enumerate(self.vocab):
                self.lookUpTable[token] = index

        except Exception as E:
            print(E)
            return -1
    

class FeedForwardNeuralNetwork(Model) :
    def __init__(self, modelName : str, previousContextSize : int = Config.previousContextSize, futureContextSize : int = Config.futureContextSize) :
        super().__init__(modelName=modelName)
        self.previousContextSize = previousContextSize
        self.futureContextSize = futureContextSize

    def train(self) :
        try:
            # Partition the dataset into minibatches 
            # for training.
            self.createMinibatchs()

            # Now that we have word embedding for each 
            # of the word, we can create inputs. 
            # self.createInputs()

        except:
            print(f"{self.modelName} Training Failed!")

    def setPreviousContextSize(self, previousContextSize : int) :
        try :
            assert isinstance(previousContextSize, int)
            self.previousContextSize = previousContextSize
        except :
            print("Previous Context Size must be an Integer!")

    def setFutureContextSize(self, futureContextSize : int) :
        try :
            assert isinstance(futureContextSize, int)
            self.futureContextSize = futureContextSize
        except :
            print("Future Context Size must be an Integer!")

    def getPreviousContextSize(self) :
        return self.previousContextSize

    def getFutureContextSize(self) :
        return self.futureContextSize

    def createMinibatchs(self) :
        pass

    def createInputs(self, dataFileName) :
        ffnnDataHandler = FeedForwardNeuralNetworkDataHandler(dataFileName=dataFileName)
        self.inputs = ffnnDataHandler.getDataPoints()


if __name__ == "__main__":
    sampleDataHandler = FeedForwardNeuralNetworkDataHandler("sample.conllu")
    sampleDataHandler.getDataPoints()
