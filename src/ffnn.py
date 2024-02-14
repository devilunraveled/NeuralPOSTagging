from typing import override

from model import Model
from dataHandler import DataHandler


class FeedForwardNeuralNetwork(Model) :
    def __init__(self, dataHandler : DataHandler, previousContextSize : int = 3, futureContextSize : int = 4) :
        super().__init__(dataHandler=dataHandler)
        self.previousContextSize = previousContextSize
        self.futureContextSize = futureContextSize

    @override
    def train(self) :
        try:
            # Partition the dataset into minibatches 
            # for training.
            self.createMinibatchs()

            # Now, we compute the word embeddings for
            # each of the word in the vocab.
            self.computeWordEmbeddings()

            # Now that we have word embedding for each 
            # of the word, we can create inputs. 
            self.createInputs()

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


    def computeWordEmbeddings(self) :
        pass

    def getWordEmbeddings(self, word) :
        pass

    def createMinibatchs(self) :
        pass

    def createInputs(self) :
        pass
