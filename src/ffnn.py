import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from src.model import Model
from src.dataHandler import DataHandler
from src.base import Evaluator, Sentence, DataPoint, OneHotEncoding, plotDevAccuracy, plotTrainLoss
from src.config import ANNConfig as Config

from alive_progress import alive_bar

EmbeddingSize = 866

class FeedForwardNeuralNetworkDataset(Dataset) :
    def __init__(self, dataPoints):
        self.dataPoints = dataPoints
        super().__init__()
    
    def __len__(self):
        return len(self.dataPoints)

    def __getitem__(self, index):
        input, label = self.dataPoints[index].getDataLabelPair()
        return torch.tensor(input, dtype=torch.float), torch.tensor(label, dtype=torch.float)

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
                tokenIndex = self.getTokenIndex.get(token, 0)
                # O corresponds to unknown tokens repsresented as `<unknown>`
                dataPointEmbedding.extend( OneHotEncoding(self.vocabSize, tokenIndex) )
                # dataPointEmbedding.extend( [1 if i == tokenIndex else 0 for i in range(self.vocabSize)] )

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

    def createRawDataPointsFromSentence(self, sentence, new = False):
        try :
            if new :
                dataPoints = []

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
            self.rawTrainDataPoints = []
            self.rawTestDataPoints = []
            self.rawValidationDataPoints = []

            self.loadDataset()

            for sentence in self.trainData.values():
                self.rawTrainDataPoints.extend(self.createRawDataPointsFromSentence(sentence))
            
            for sentence in self.testData.values():
                self.rawTestDataPoints.extend(self.createRawDataPointsFromSentence(sentence))

            for sentence in self.validationData.values():
                self.rawValidationDataPoints.extend(self.createRawDataPointsFromSentence(sentence))

            return self.rawTrainDataPoints, self.rawTestDataPoints, self.rawValidationDataPoints
        except Exception as e:
            print(e)
            return []

    def getDataPoints(self):
        try :
            self.createRawDataPoints()
            self.createLookUpTable()

            self.trainDataPoints = []
            self.testDataPoints = []
            self.validationDataPoints = []

            for rawTrainDataPoint in self.rawTrainDataPoints:
                newDataPoint = FeedForwardNeuralNetworkDataPoint( dataPoint=rawTrainDataPoint, lookUp=self.lookUpTable, position=self.previousContextSize, vocabSize=len(self.vocab))
                self.trainDataPoints.append( newDataPoint )
            
            for rawTestDataPoint in self.rawTestDataPoints:
                newDataPoint = FeedForwardNeuralNetworkDataPoint( dataPoint=rawTestDataPoint, lookUp=self.lookUpTable, position=self.previousContextSize, vocabSize=len(self.vocab))
                self.testDataPoints.append( newDataPoint )
            
            for rawValidationDataPoint in self.rawValidationDataPoints:
                newDataPoint = FeedForwardNeuralNetworkDataPoint( dataPoint=rawValidationDataPoint, lookUp=self.lookUpTable, position=self.previousContextSize, vocabSize=len(self.vocab))
                self.validationDataPoints.append( newDataPoint )

            return self.trainDataPoints

        except Exception as e:
            print(e)
            return []



class FeedForwardNeuralNetwork(nn.Module, Model) :
    def __init__(self, modelName : str, previousContextSize : int = Config.previousContextSize, 
                 futureContextSize : int = Config.futureContextSize, 
                 hiddenLayerSize : list = Config.hiddenLayers, device = None, startOver = False, batchSize : int = Config.miniBatchSize) :
        
        if device is None :
            device = 'cuda'

        self.device = torch.device(device)
        print(f"Using {self.device} for FFNN.")
        
        self.previousContextSize = previousContextSize
        self.futureContextSize = futureContextSize
        self.batchSize = batchSize
        self.hiddenLayerSize = hiddenLayerSize
        self.numHiddenLayers = len(self.hiddenLayerSize)
        
        self.numClasses = len(Config.classLabels)
        self.inputSize = EmbeddingSize*(self.previousContextSize + self.futureContextSize + 1)
        self.modelLoss = []
        self.modelAccuracyDev = []
        
        modelName += f"_p={previousContextSize}_s={futureContextSize}_n={self.numHiddenLayers}"
        for i in range(self.numHiddenLayers) :
            modelName += f"_{hiddenLayerSize[i]}"

        Model.__init__(self,modelName=modelName)
        nn.Module.__init__(self)
        
        if not startOver :
            try :
                resp = self.loadModel()
                if resp == 1 :
                    return 
            except :
                pass

        self.relu = nn.ReLU()
        self.layer = nn.ModuleList()
        
        for i in range (self.numHiddenLayers + 1) :
            if i == 0 :
                self.layer.append(nn.Linear(self.inputSize, self.hiddenLayerSize[0]))
            elif i == self.numHiddenLayers :
                self.layer.append(nn.Linear(self.hiddenLayerSize[-1], self.numClasses))
            else :
                self.layer.append(nn.Linear(self.hiddenLayerSize[i-1], self.hiddenLayerSize[i]))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.trained = False
        
        self.to(self.device)
    
    def setNNParams(self):
        self.relu = nn.ReLU()
        self.layer = nn.ModuleList()
        
        for i in range (self.numHiddenLayers + 1) :
            if i == 0 :
                self.layer.append(nn.Linear(self.inputSize, self.hiddenLayerSize[0]))
            elif i == self.numHiddenLayers :
                self.layer.append(nn.Linear(self.hiddenLayerSize[-1], self.numClasses))
            else :
                self.layer.append(nn.Linear(self.hiddenLayerSize[i-1], self.hiddenLayerSize[i]))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
        for i in range(self.numHiddenLayers + 1):
            x = self.layer[i](x)
            if i != self.numHiddenLayers :
                x = self.relu(x)
        return x

    def trainModel(self, numEpochs = Config.epochs):
        try:
            if self.trained :
                return 
            
            if not hasattr(self, 'dataHandler') :
                self.prepareData()
            
            for epoch in range(numEpochs):
                runningLoss = 0.0
                with alive_bar(len(self.trainData)) as bar:
                    for inputs, labels in self.trainData:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        # Forward pass
                        outputs = self(inputs)

                        loss = self.criterion(outputs, labels)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # Print statistics
                        runningLoss += loss.item()
                        bar()
                
                # Print average loss for the epoch
                avg_loss = runningLoss / len(self.trainData)
                self.modelLoss.append(avg_loss)
                print(f'Epoch [{epoch+1}/{numEpochs}], Average Loss: {avg_loss:.4f}')

                validationMetric = self.evaluateModel(validation = True)
                self.modelAccuracyDev.append(validationMetric)

            print('Training completed successfully')
            self.trained = True
            self.saveModel()
            return 1
        except Exception as e:
            raise(e)
        
    def evaluateModel(self, validation = False, test = False):
        try :
            if validation is True :
                data = self.validationData
            elif test is True :
                data = self.testData
            else :
                data = self.trainData

            accuracy = 0
            batchSize = 0
            allPredictions = []
            allActuals = []
            
            with torch.no_grad() :
                with alive_bar(len(data)) as bar:
                    for X,Y in data :
                        if batchSize == 0 :
                            batchSize = X.shape[0]
                        X, Y = X.to(self.device), Y.to(self.device)
                        outputs = self(X)

                        predicted = torch.argmax(outputs, dim = 1)
                        actual = torch.argmax(Y, dim = 1)
                        

                        allPredictions.extend([Config.classes[predictedValue] for predictedValue in predicted.flatten().tolist()])
                        allActuals.extend([Config.classes[actualValue] for actualValue in actual.flatten().tolist()])

                        accuracy += (predicted == actual).sum().item()
                        bar()
            
            evaluation = Evaluator(allPredictions, allActuals)
            
            print(evaluation())
            if data == self.testData :
                evaluation.plot_confusion_matrix(fileName = self.modelName)
            return evaluation
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise(e)
   
    def completeEvaluation(self):
        try:
            results = {}
            # results['Train'] = self.evaluateModel()
            results['Validation'] = self.evaluateModel(validation = True)
            results['Test'] = self.evaluateModel(test = True)
            
            plotDevAccuracy(fileName = self.modelName, devAccuracy = self.modelAccuracyDev)
            plotTrainLoss(trainLoss = self.modelLoss, fileName = self.modelName)

            return results
        except Exception as e:
            raise(e)
            return {}
    
    def prepareData(self):
        try :
            if hasattr(self, 'dataHandler') :
                return 1

            self.dataHandler = FeedForwardNeuralNetworkDataHandler("en_atis-ud", previousContextSize=self.previousContextSize, futureContextSize=self.futureContextSize)
            self.dataHandler.getDataPoints()
            
            self.trainDataSet = FeedForwardNeuralNetworkDataset(self.dataHandler.trainDataPoints)
            self.testDataSet = FeedForwardNeuralNetworkDataset(self.dataHandler.testDataPoints)
            self.validationDataSet = FeedForwardNeuralNetworkDataset(self.dataHandler.validationDataPoints)

            self.trainData = DataLoader(self.trainDataSet, batch_size=self.batchSize, shuffle=True)
            self.testData = DataLoader(self.testDataSet, batch_size=self.batchSize, shuffle=False)
            self.validationData = DataLoader(self.validationDataSet, batch_size=self.batchSize, shuffle=False)
        except Exception as e:
            print(e)
            return
        
    def inference(self, sentence):
        try :
            self.prepareData()
            
            if self.trained is False :
                self.trainModel(numEpochs = Config.epochs)
            
            information = {}
            information['sent_id'] = "1.inference"
            information['text'] = sentence
            information['POS'] = ['X' for _ in range(len(sentence.split(" ")))]

            sentence = Sentence(information)
            sentenceRawDataPoints = self.dataHandler.createRawDataPointsFromSentence(sentence)
            
            dataPoints = []

            for rawDataPoint in sentenceRawDataPoints :
                dataPoints.append(FeedForwardNeuralNetworkDataPoint(rawDataPoint, self.dataHandler.lookUpTable, self.previousContextSize, len(self.dataHandler.vocab)))

            inferenceDataSet = FeedForwardNeuralNetworkDataset(dataPoints)
            inferenceData = DataLoader(inferenceDataSet, shuffle=False)
            
            prediction = []
            with torch.no_grad():
                for X,_ in inferenceData :
                    X = X.to(self.device)
                    outputs = self(X)
                    predicted = torch.argmax(outputs, dim = 1)
                    prediction.append(Config.classes[predicted[0]])

            information['POS'] = prediction
            
            sentence = Sentence(information)

            print(sentence)

        except Exception as e:
            raise(e)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedForwardNeuralNetwork("SimpleANN", device = device)
    model.prepareData()
    model.trainModel(numEpochs = Config.epochs)
    results = model.completeEvaluation()
    
    # print(f"{'Parition':<10}|{'Accuracy':<9}(%)")
    # for key in results :
    #     print(f"{key:<10}|{results[key]*100:<9.2f}%")

    # sentence = input("Enter Sentence : ")
    # model.inference(sentence)
