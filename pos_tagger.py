import argparse

from src.ffnn import FeedForwardNeuralNetwork
from src.rnn import ReccurentNeuralNetwork

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="ffnn", help="Inference From FFNN")
    parser.add_argument("-r", dest="rnn", help="Inference From RNN")
    parser.add_argument("-b", dest="brnn", help="Inference From Biderectional RNN")

    arguments = parser.parse_args()

    if arguments.ffnn is not None:
        ffnn = FeedForwardNeuralNetwork(modelName = "ANN")
        print("Using FFNN for Inference...")
        ffnn.inference(arguments.ffnn)

    if arguments.rnn is not None:
        rnn = ReccurentNeuralNetwork(modelName = "RNN")
        print("Using RNN for Inference...")
        rnn.inference(arguments.rnn)

    if arguments.brnn is not None:
        brnn = ReccurentNeuralNetwork(modelName = "RNN" , bidirectionality = True)
        print("Using Biderectional RNN for Inference...")
        brnn.inference(arguments.brnn)

main()
