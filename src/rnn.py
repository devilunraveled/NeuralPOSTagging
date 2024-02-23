"""
    This file implements the RNN model
    for POS Tagging Class. We can obtain the
    data from the DataHandler Class
"""

from alive_progress import alive_bar

from src.model import Model
from src.config import RNNConfig as Config
from src.base import Sentence, DataPoint,OneHotEncoding, Evaluator, plotDevAccuracy, plotTrainLoss
from src.dataHandler import DataHandler

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch

from src.ffnn import EmbeddingSize

class ReccurentNeuralNetworkDataset(Dataset):
    def __init__(self, dataPoints) -> None:
        self.dataPoints = dataPoints
        super().__init__()

    def __len__(self):
        return len(self.dataPoints)

    def __getitem__(self, index):
        input, label = self.dataPoints[index].getDataLabelPair()
        return torch.tensor(input, dtype=torch.float), torch.tensor(label, dtype=torch.float)

class ReccurentNeuralNetworkDataPoint(DataPoint):
    def __init__(self, dataPoint, lookUp : dict, vocabSize) :
        self.dataPoint = [token[0] for token in dataPoint], [token[1] for token in dataPoint]
        self.getTokenIndex = lookUp
        self.vocabSize = vocabSize

    def getDataEmbedding(self):
        try :
            dataPointEmbedding = []

            for token in self.dataPoint[0]:
                tokenIndex = self.getTokenIndex.get(token, 0)
                dataPointEmbedding.append( OneHotEncoding(self.vocabSize, tokenIndex) )

            return dataPointEmbedding
        except Exception as e:
            print(e)
            return []

    def getLabelEmbedding(self):
        try:
            labelEmbedding = []

            for label in self.dataPoint[1]:
                labelEmbedding.append( OneHotEncoding(Config.classLabels.__len__(), Config.classLabels[label]) )

            return labelEmbedding
        except Exception as e:
            print(e)
            return []

    def getDataLabelPair(self):
        return self.getDataEmbedding(), self.getLabelEmbedding()

class ReccurentNeuralNetworkDataHandler(DataHandler):
    def __init__(self, dataFileName : str) -> None:
        super().__init__(dataFileName=dataFileName)

    def createDataPointFromSentence(self, sentence : Sentence) :
        try :
            assert isinstance(sentence, Sentence)
            dataPoint = ReccurentNeuralNetworkDataPoint(dataPoint=sentence.getTokensWithLabels(), lookUp=self.lookUpTable, vocabSize=len(self.vocab))
            return dataPoint
        except Exception as e:
            print(e)
            return []
    
    def getDataPoints(self):
        try :
            self.loadDataset()
            self.createLookUpTable()

            self.trainDataPoints = [ self.createDataPointFromSentence(sentence) for sentence in self.trainData.values() ]
            self.validationDataPoints = [ self.createDataPointFromSentence(sentence) for sentence in self.validationData.values() ]
            self.testDataPoints = [ self.createDataPointFromSentence(sentence) for sentence in self.testData.values() ]
            
            return self.trainDataPoints
        except Exception as e:
            print(e)
            return []

class ReccurentNeuralNetwork(nn.Module, Model):
    def __init__(self, modelName : str, startOver : bool = False, stackSize : int = Config.stackSize, 
                 inputSize = EmbeddingSize, device = None, nonLinearity : str = 'tanh', bidirectionality = False, 
                 hiddenLayers : list = Config.hiddenLayers, hiddenStateSize : int = Config.hiddenStateSize, 
                 miniBatchSize : int = Config.miniBatchSize ) :
        
        ### Static Variables, can be used for model naming.
        self.numClasses = len(Config.classLabels)
        self.hiddenLayerSize = hiddenLayers
        self.numHiddenLayers = len(self.hiddenLayerSize)
        self.inputSize = inputSize
        self.stackSize = stackSize
        self.hiddenStateSize = hiddenStateSize
        self.nonLinearity = nonLinearity
        self.bidirectionality = bidirectionality
        self.miniBatchSize = miniBatchSize
        
        self.modelLoss = []
        self.modelDevAccuracy = []

        if device is None :
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using {self.device}")
        else :
            self.device = device

        ### Constructing Model's name
        modelName += f"_n={stackSize}"
        modelName += f"_h={self.hiddenStateSize}"
        
        if bidirectionality :
            modelName += "_bidirectional"

        modelName += "_nonLinearity="+nonLinearity
        
        modelName += f"_layers={self.numHiddenLayers}"
        for layerSize in self.hiddenLayerSize :
            modelName += f"_l={layerSize}"

        Model.__init__(self,modelName=modelName)
        nn.Module.__init__(self)

        if not startOver :
            try :
                resp = self.loadModel()
                if resp == 1 :
                    return
            except :
                pass


        self.RNN = nn.RNN(input_size=self.inputSize, device = self.device, hidden_size=self.hiddenStateSize, 
                            num_layers=self.stackSize, nonlinearity=self.nonLinearity,bidirectional=self.bidirectionality, 
                            batch_first=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.003)
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList()
        for i in range (self.numHiddenLayers + 1) :
            if i == 0 :
                self.layers.append(nn.Linear(self.hiddenStateSize*( 1 + self.bidirectionality), self.hiddenLayerSize[i]))
            elif i == self.numHiddenLayers :
                self.layers.append(nn.Linear(self.hiddenLayerSize[-1], self.numClasses))
            else :
                self.layers.append(nn.Linear(self.hiddenLayerSize[i-1], self.hiddenLayerSize[i]))
        self.softmax = nn.Softmax(dim=1)

        self.trained = False

        self.to(self.device)
    
    def initializeHiddenState(self, batchSize):
        return torch.zeros(self.stackSize*(1+self.bidirectionality), batchSize, self.hiddenStateSize).to(self.device)

    def forward(self, x):
        batchSize = x.shape[0]
        self.hiddenState = self.initializeHiddenState(batchSize).to(self.device)
        x, _ = self.RNN(x, self.hiddenState)
       
        for i in range(self.numHiddenLayers + 1):
            x = self.layers[i](x)
            if i != self.numHiddenLayers :
                x = self.relu(x)
            else :
                x = self.softmax(x)
        return x

    def trainModel(self, numEpochs : int = 10):
        try :
            if self.trained :
                return
            
            for epoch in range(numEpochs) :
                runningLoss = 0.0
                with alive_bar(len(self.trainData)) as bar :
                    for inputs, labels in self.trainData :
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self(inputs)        
                        # print(f"{outputs.shape=}, {labels.shape=}")
                        loss = self.criterion(outputs, labels)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        runningLoss += loss.item()
                        bar()

                averageLoss = runningLoss / len(self.trainData)
                self.modelLoss.append(averageLoss)
                print(f"Epoch : [{epoch + 1} | {numEpochs}] | Loss : {averageLoss:.4f}")
            
                validationMetric = self.evaluateModel(validation = True)
                self.modelDevAccuracy.append(validationMetric)

            print('Training completed successfully')
            self.trained = True
            self.saveModel()
            return 1
        except Exception:
            import traceback
            print(traceback.format_exc())
            return None
    def prepareData(self) :
        try :
            if hasattr(self, 'dataHandler') :
                return

            self.dataHandler = ReccurentNeuralNetworkDataHandler("en_atis-ud")
            self.dataHandler.getDataPoints()
            
            print(f"Got the dataPoints...")

            self.trainDataSet = ReccurentNeuralNetworkDataset(self.dataHandler.trainDataPoints)
            self.validationDataSet = ReccurentNeuralNetworkDataset(self.dataHandler.validationDataPoints)
            self.testDataSet = ReccurentNeuralNetworkDataset(self.dataHandler.testDataPoints)
            
            print("Initialized Datasets...")
            print("Loading the data")
            
            self.trainData = DataLoader(self.trainDataSet, batch_size=self.miniBatchSize, shuffle=True, collate_fn=self.customCollate)
            self.validationData = DataLoader(self.validationDataSet, batch_size=self.miniBatchSize, shuffle = False, collate_fn=self.customCollate)
            self.testData = DataLoader(self.testDataSet, batch_size=self.miniBatchSize, shuffle = False, collate_fn=self.customCollate)

            print(f"Data Loaded Successfully")
        except Exception as e:
            print(e)
            return

    def evaluateModel(self, validation = False, test = False) :
        try :
            if validation is True :
                data = self.validationData
            elif test is True :
                data = self.testData
            else :
                data = self.trainData

            batchSize = 0
            
            self.RNN.flatten_parameters()

            allPredictions = []
            allActuals = []
            with alive_bar(len(data)) as bar :
                with torch.no_grad():
                    for inputs, labels in data :
                        if batchSize == 0 :
                            batchSize = inputs.shape[0]
                        
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        batchSize = inputs.shape[0]

                        outputs = self(inputs)
                        
                        for i, label in enumerate(labels):
                            for j, oneHotVector in enumerate(label):
                                maxIndex = torch.argmax(oneHotVector).item()
                                maxValue = label[j][int(maxIndex)]

                                if maxValue.item() == 0.0 :
                                    continue

                                allActuals.append(Config.classes[int(maxIndex)] )
                                allPredictions.append(Config.classes[int(torch.argmax(outputs[i][j]))])

                        bar()

            evaluation = Evaluator(allPredictions, allActuals)

            print(evaluation())
            if data == self.testData :
                evaluation.plot_confusion_matrix(fileName = self.modelName)
            return evaluation

        except Exception:
            import traceback
            print(traceback.format_exc())
            return

    def completeEvaluation(self):
        try:
            results = {}
            results['Train'] = self.evaluateModel()
            results['Validation'] = self.evaluateModel(validation = True)
            results['Test'] = self.evaluateModel(test = True)
            
            plotDevAccuracy(fileName = self.modelName, devAccuracy = self.modelDevAccuracy)
            plotTrainLoss(fileName = self.modelName, trainLoss = self.modelLoss)

            return results
        except Exception as e:
            print(e)
            return {}
        
    def inference(self, sentence):
        try :
            self.prepareData()
            if self.trained is False :
                self.trainModel(numEpochs = Config.epochs)
            
            self.RNN.flatten_parameters()
            information = {}
            information['sent_id'] = "1.inference"
            information['text'] = sentence
            information['POS'] = ['X' for _ in range(len(sentence.split(" ")))]

            sentence = Sentence(information)
            
            with torch.no_grad():
                dataPoint = self.dataHandler.createDataPointFromSentence(sentence)
                
                inferenceDataSet = ReccurentNeuralNetworkDataset([dataPoint])
                inferenceData = DataLoader(inferenceDataSet, shuffle=False)
                prediction = []
                for X,_ in inferenceData :
                    X = X.to(self.device)
                    outputs = self(X)
                    predicted = torch.argmax(outputs, dim = -1)
                    prediction.extend([Config.classes[pred] for pred in predicted.flatten().tolist()])

            information['POS'] = prediction
            
            sentence = Sentence(information)

            print(sentence)

        except Exception as e:
            raise(e)


    def customCollate(self, batch):
        # Sort the batch by sequence length in descending order
        batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=False)
        # Extract sequences and labels from the batch
        sequences, labels = zip(*batch)
        # Pad sequences to the length of the longest sequence
        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Pad labels to the length of the longest sequence
        padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return padded_sequences, padded_labels

if __name__ == "__main__":
    model = ReccurentNeuralNetwork(modelName="SimpleRNN", startOver=False, bidirectionality = True)
    model.prepareData()
    model.trainModel(numEpochs = Config.epochs)
    results = model.completeEvaluation()
    
    # sentence = "I am a flight"
    # sentence = input("Enter Sentence : ")
    # model.inference(sentence)
    # print(results)
