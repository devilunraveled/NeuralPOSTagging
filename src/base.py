from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config

class Color :
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Sentence :
    def __init__(self, information : dict):
        self.id = information.get("sent_id", None)
        self.partition = self.id.split(".")[-1]
        self.original = information.get("text", "")
        self.POStags = information.get("POS", [])
    
    def getTokensWithLabels(self):
        return list(zip(self.original.split(" "), self.POStags))

    def __str__(self):
        text = ""
        # text = f"{Color.YELLOW}Sentence ID : {self.id}{Color.END}\n"
        for token,POStag in zip(self.original.split(" "), self.POStags):
            text += f"{Color.CYAN}{token:<12.10} {Color.BOLD}{Color.GREEN}{POStag:<8.6}{Color.END} \n"

        return text

class Evaluator :
    def __init__(self, prediction, groundTruth) :
        self.prediction = prediction
        self.groundTruth = groundTruth

    def getConfusionMatrix(self) :
        return confusion_matrix(self.groundTruth, self.prediction)

    def getAccuracy(self) :
        return accuracy_score(self.groundTruth, self.prediction, normalize = True)

    def getPrecision(self, average = None) :
        average = "weighted" if average is None else average
        return precision_score(self.groundTruth, self.prediction, average = average, zero_division=0.0)

    def getF1Score(self, average = None) :
        average = "weighted" if average is None else average
        return f1_score(self.groundTruth, self.prediction, average = average, zero_division=0.0)
    
    def getRecall(self, average = None) :
        average = "weighted" if average is None else average
        return recall_score(self.groundTruth, self.prediction, average = average, zero_division=0.0)

    def plot_confusion_matrix(self, normalize = False, fileName : str = "Model", showPlot : bool = False ):
        try :
            cm = self.getConfusionMatrix()
            fmt = 'd'
            if normalize:
                # Normalize the confusion matrix
                cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]*(0.001))
                fmt = '.2f'  # Format for displaying normalized values

            fig = plt.figure(figsize=(10, 8))

            uniqueLabels = np.unique(self.groundTruth + self.prediction)
            classLabels = [label for label in uniqueLabels]

            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classLabels, yticklabels=classLabels, vmin=0, vmax=200)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.savefig(f'{Config.plotSavePath}CM_{fileName}.svg', bbox_inches='tight')
            if showPlot :
                plt.show()
            plt.close(fig)
        except RuntimeError as e :
            print(e)

    def __call__(self):
        metrics = {
            "Accuracy" : self.getAccuracy(),
            "Precision" : self.getPrecision(),
            "F1Score" : self.getF1Score(),
            "Recall" : self.getRecall(),
        }
        
        return metrics
    
class DataPoint :
    def __init__(self, dataPoint):
        self.dataPoint = [token[0] for token in dataPoint]

    def __str__(self):
        return str(self.dataPoint)


def OneHotEncoding(totalSize, index) :
    oneHotEncoding = [0] * totalSize
    if index < totalSize :
        oneHotEncoding[index] = 1
    else :
        print("Error in OneHotEncoding function")
    return oneHotEncoding

def plotDevAccuracy(devAccuracy : list, fileName : str = "Model") :
    devAccuracy = [ met()['Accuracy']*100 for met in devAccuracy ]
    fig = plt.figure()
    plt.plot(devAccuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f'{Config.plotSavePath}DevAccuracy_{fileName}.svg', bbox_inches='tight')
    plt.close(fig)

def plotTrainLoss(trainLoss : list, fileName : str = "Model") :
    fig = plt.figure()
    plt.plot(trainLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f'{Config.plotSavePath}TrainLoss_{fileName}.svg', bbox_inches='tight')
    plt.close(fig)

