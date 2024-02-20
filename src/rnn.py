"""
    This file implements the RNN model
    for POS Tagging Class. We can obtain the
    data from the DataHandler Class
"""

from alive_progress import alive_bar
from model import Model
from config import Config
from base import Sentence, DataPoint,OneHotEncoding, Evaluator
from dataHandler import DataHandler

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch

from ffnn import EmbeddingSize

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
        self.dataPoint = [token[0] for token in dataPoint],[OneHotEncoding(len(Config.classLabels),Config.classLabels[label[1]]) for label in dataPoint]
        self.getTokenIndex = lookUp
        self.vocabSize = vocabSize

    def getEmbedding(self):
        try :
            dataPointEmbedding = []

            for token in self.dataPoint[0]:
                tokenIndex = self.getTokenIndex.get(token, 0)
                dataPointEmbedding.extend( OneHotEncoding(self.vocabSize, tokenIndex) )

            return dataPointEmbedding
        except Exception as e:
            print(e)
            return []

    def getDataLabelPair(self):
        return self.getEmbedding(), self.dataPoint[1]

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
    def __init__(self, modelName : str, recurrenceDepth : int = Config.recurrenceDepth, startOver : bool = False, stackSize : int = Config.stackSize, inputSize = EmbeddingSize ) :
        modelName += f"_n={recurrenceDepth}"

        Model.__init__(self,modelName=modelName)
        nn.Module.__init__(self)

        if not startOver :
            try :
                resp = self.loadModel()
                if resp == 1 :
                    return
            except :
                pass

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.inputSize = inputSize
        self.stackSize = stackSize
        self.hiddenStateSize = Config.hiddenStateSize
        
        self.RNN = nn.RNN(input_size=self.inputSize, device = self.device, hidden_size=self.hiddenStateSize, batch_first=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        self.trained = False
    
    def forward(self, x):
        batchSize = x.shape[0]
        hiddenState = self.init_hidden(batchSize)
        out, _ = self.RNN(x, hiddenState)
        return out

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
                        loss = self.criterion(outputs, labels)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        runningLoss += loss.item()
                        bar()

                averageLoss = runningLoss / len(self.trainData)
                print(f"Epoch : [{epoch + 1/numEpochs}] | Loss : {averageLoss:.4f}")
            
                if epoch > 0 and epoch % Config.validationFrequency == 1 :
                    self.evaluateModel(validation = True)

            print('Training completed successfully')
            self.trained = True
            self.saveModel()
            return 1
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise(e)
        
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
            
            print(f"{len(self.trainDataSet)=} | {len(self.validationDataSet)=} | {len(self.testDataSet)=}")

            self.trainData = DataLoader(self.trainDataSet, batch_size=Config.miniBatchSize, shuffle=True, collate_fn=self.customCollate)
            self.validationData = DataLoader(self.validationDataSet, batch_size=Config.miniBatchSize, shuffle = False, collate_fn=self.customCollate)
            self.testData = DataLoader(self.testDataSet, batch_size=Config.miniBatchSize, shuffle = False, collate_fn=self.customCollate)

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
            allPredictions = []
            allActuals = []

            with alive_bar(len(data)) as bar :
                for inputs, labels in data :
                    if batchSize == 0 :
                        batchSize = inputs.shape[0]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batchSize = inputs.shape[0]

                    outputs = self(inputs)

                    predicted = torch.argmax(outputs, dim=1)
                    actual = torch.argmax(labels, dim=1)

                    allPredictions.extend([Config.classes[predictedValue] for predictedValue in predicted.flatten().tolist()])
                    allActuals.extend([Config.classes[actualValue] for actualValue in actual.flatten().tolist()])
                    bar()

            evaluation = Evaluator(allPredictions, allActuals)

            evaluation.plot_confusion_matrix()
            return evaluation

        except Exception as e :
            print(e)
            return

    def completeEvaluation(self):
        try:
            results = {}
            results['Train'] = self.evaluateModel()
            results['Validation'] = self.evaluateModel(validation = True)
            results['Test'] = self.evaluateModel(test = True)

            return results
        except Exception as e:
            print(e)
            return {}
        
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
            dataPoint = self.dataHandler.createDataPointFromSentence(sentence)
            
            inferenceDataSet = ReccurentNeuralNetworkDataset(dataPoint)
            inferenceData = DataLoader(inferenceDataSet, shuffle=False)
            
            prediction = []
            with torch.no_grad():
                for X,_ in inferenceData :
                    X = X.to(self.device)
                    outputs = self(X)
                    predicted = torch.argmax(outputs, dim = 1)
                    prediction.append(Config.classes[predicted] for predicted in predicted.flatten().tolist())

            information['POS'] = prediction
            
            sentence = Sentence(information)

            print(sentence)

        except Exception as e:
            raise(e)


    def customCollate(self, batch):
        # Sort the batch by sequence length in descending order
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Extract sequences and labels from the batch
        sequences, labels = zip(*batch)
        # Pad sequences to the length of the longest sequence
        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Convert labels to a tensor
        labels = torch.stack(labels)
        return padded_sequences, labels

if __name__ == "__main__":
    model = ReccurentNeuralNetwork(modelName="SimpleRNN")
    model.prepareData()
    model.trainModel(numEpochs = Config.epochs)
    results = model.completeEvaluation()

    print(results)
