import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from model import Model
from dataHandler import DataHandler
from base import Sentence, DataPoint, OneHotEncoding
from config import Config

from alive_progress import alive_bar

class FeedForwardNeuralNetworkDataset(Dataset) :
    def __init__(self, dataPoints):
        self.dataPoints = dataPoints
        super().__init__()
    
    def __len__(self):
        return len(self.dataPoints)

    def __getitem__(self, index):
        input, label = self.dataPoints[index].getDataLabelPair()
        return torch.tensor(input, dtype=torch.float), torch.tensor(label, dtype=torch.float32)

class FeedForwardNeuralNetworkDataPoint(DataPoint) :
    def __init__(self, dataPoint, lookUp : dict, position, vocabSize) :
        self.dataPoint = [token[0] for token in dataPoint], OneHotEncoding(Config.classLabels.__len__(), Config.classLabels[dataPoint[position][1]])
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

    def getDataLabelPair(self):
        return self.getEmbedding(), self.dataPoint[1]

    def __str__(self):
        description = f"RawDataPoint : {self.dataPoint}\n"
        description += f"Target : {self.dataPoint[0][self.position]}\n"
        description += f"DataLabelPair : {self.getDataLabelPair()}\n"

        return description

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
                self.dataPoint.append( newDataPoint )

            return self.dataPoint

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


class FeedForwardNeuralNetwork(nn.Module, Model) :
    def __init__(self, modelName : str, previousContextSize : int = Config.previousContextSize, 
                 futureContextSize : int = Config.futureContextSize, numHiddenLayers : int = 1, 
                 hiddenLayerSize : list = [64], vocabSize : int = 866) :
        
        Model.__init__(self,modelName=modelName)
        nn.Module.__init__(self)
        
        self.previousContextSize = previousContextSize
        self.futureContextSize = futureContextSize
        
        self.numHiddenLayers = numHiddenLayers
        self.hiddenLayerSize = hiddenLayerSize

        self.numClasses = len(Config.classLabels)
        self.inputSize = vocabSize*(self.previousContextSize + self.futureContextSize + 1)

        self.fc1 = nn.Linear(self.inputSize, self.hiddenLayerSize[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hiddenLayerSize[0], self.numClasses)
        self.softmax = nn.Softmax(dim = 1)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
    
    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def trainModel(self, dataLoader, numEpochs):
        try:
            try :
                resp = self.loadModel()
                if resp == 1 :
                    return 1
            except :
                pass

            with alive_bar(len(dataLoader)*numEpochs) as bar:
                for epoch in range(numEpochs):
                    runningLoss = 0.0
                    for inputs, labels in dataLoader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        # Zero the parameter gradients
                        self.optimizer.zero_grad()

                        # Forward pass
                        outputs = self(inputs)
                        loss = self.criterion(outputs, labels)

                        # Backward pass and optimization
                        loss.backward()
                        self.optimizer.step()

                        # Print statistics
                        runningLoss += loss.item()
                        bar()
                    # Print average loss for the epoch
                    avg_loss = runningLoss / len(dataLoader)
                    print(f'Epoch [{epoch+1}/{numEpochs}], Average Loss: {avg_loss:.4f}')

            print('Training completed successfully')
            self.saveModel()
            return 1
        except Exception as e:
            raise(e)
            return -1
        
    def evaluateModel(self, data):
        try :
            # X = torch.tensor(data[0], dtype=torch.float32)
            # Y = torch.tensor(data[1], dtype=torch.float32)
            accuracy = 0 
            for X,Y in data :
                X, Y = X.to(device), Y.to(device)
                outputs = self(X)
                predicted = torch.argmax(outputs, 1)
                accuracy += (predicted == Y).sum().item() / Y.size(0)
                
            print(f'Accuracy: {accuracy/len(data):.4f}')
        except Exception as e:
            raise(e)
            return -1

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=}")

    trainDataHandler = FeedForwardNeuralNetworkDataHandler("en_atis-ud-train.conllu")
    trainDataSet = FeedForwardNeuralNetworkDataset(trainDataHandler.getDataPoints())
    
    testDataHandler = FeedForwardNeuralNetworkDataHandler("en_atis-ud-train.conllu")
    testDataSet = FeedForwardNeuralNetworkDataset(testDataHandler.getDataPoints())

    miniBatchSize = Config.miniBatchSize
    trainDataLoader = DataLoader(trainDataSet, batch_size=miniBatchSize, shuffle=True)
    testDataLoader = DataLoader(testDataSet, shuffle=True)

    model = FeedForwardNeuralNetwork("sampleModel").to(device)
    model.trainModel(dataLoader = trainDataLoader, numEpochs = Config.epochs)
    model.evaluateModel(testDataLoader)
