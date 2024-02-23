from src.ffnn import FeedForwardNeuralNetwork
from src.rnn import ReccurentNeuralNetwork

class PerformanceEvaluation:
    def __init__(self, parameters : list[dict]) -> None:
        self.parameters = parameters
        self.evaluations = {}
        self.models = []

    def runEvaluations(self) -> None:
        for params in self.parameters :
            model = ReccurentNeuralNetwork(**params)
            self.models.append(model)
            model.prepareData()
            model.trainModel()
            self.evaluations[model.modelName] = model.completeEvaluation()

def getParamsFFNN():
    params = []
    
    possibleS = [0,1,2,3,4]
    possibleP = [0,1,2,3,4]
    possibleHiddenLayerSizes = [ [32], [64], [128],
                             [32, 32], [64, 64],
                             [64, 32], [128, 64],
                             [128, 64, 32] ]

    for s in possibleS :
        for p in possibleP :
            for hiddenLayerSize in possibleHiddenLayerSizes :
                params.append({
                    "previousContextSize" : p,
                    "futureContextSize" : s,
                    "hiddenLayerSize" : hiddenLayerSize,
                    "modelName" : f"EvaluationFFNN-p={p}_s={s}_n={len(hiddenLayerSize)}",
                })

    return params

def getParamsRNN():
    params = []

    possibleStackSize = [1,2,3]
    possibleHiddenLayerSizes = [ [32], [64], [128],
                             [32, 32], [64, 64],
                             [64, 32], [128, 64],
                             [128, 64, 32] ]
    possibleHiddenStateSizes = [ 64, 128, 256 ]
    possibleBidirectionality = [True, False]
    
    for stackSize in possibleStackSize :
        for hiddenLayerSize in possibleHiddenLayerSizes :
            for hiddenStateSize in possibleHiddenStateSizes :
                for bidirectionality in possibleBidirectionality :
                    params.append({
                        "stackSize" : stackSize,
                        "hiddenLayers" : hiddenLayerSize,
                        "hiddenStateSize" : hiddenStateSize,
                        "bidirectionality" : bidirectionality,
                        "modelName" : f"EvaluationRNN-s={stackSize}_h={len(hiddenLayerSize)}_b={bidirectionality}_l={hiddenStateSize}",
                    })

    return params

if __name__ == "__main__":
    # ffnnEval = PerformanceEvaluation(getParamsFFNN())
    # ffnnEval.runEvaluations()

    rnnEval = PerformanceEvaluation(getParamsRNN())
    rnnEval.runEvaluations()
