from src.ffnn import FeedForwardNeuralNetwork
from src.model import Model

class PerformanceEvaluation:
    def __init__(self, model : Model, parameters : list[dict]) -> None:
        assert isinstance(model, Model)
        self.model = model
        self.parameters = parameters
        self.evaluations = {}

    def runEvaluations(self) -> None:
        for params in self.parameters :
            self.model.setParams(**params)
            self.model.setNNParams()
            self.model.trainModel()
            self.evaluations[self.model.modelName] = self.model.completeEvaluation()


def getParamsFFNN():
    params = []

    params.append({
        "previousContextSize" : 1,
        "futureContextSize" : 1,
        "numHiddenLayers" : 1,
        "hiddenLayerSize" : [64],
        "modelName" : "EvaluationFFNN-1",
    })

    params.append({
        "previousContextSize" : 2,
        "futureContextSize" : 2,
        "numHiddenLayers" : 1,
        "hiddenLayerSize" : [64],
        "modelName" : "EvaluationFFNN-2",
    })

    params.append({
        "previousContextSize" : 3,
        "futureContextSize" : 3,
        "numHiddenLayers" : 1,
        "hiddenLayerSize" : [64],
        "modelName" : "EvaluationFFNN-3",
    })
    
    return params

if __name__ == "__main__":
    sampleFNN = FeedForwardNeuralNetwork("EvaluationANN")

    ffnnEval = PerformanceEvaluation(FeedForwardNeuralNetwork("ffnn"), getParamsFFNN())
    ffnnEval.runEvaluations()
