class Config :
    modelSavePath = "../checkpoints/"
    fileFormat = ".conllu"
    trainTag = "-train"
    testTag = "-test"
    validationTag = "-dev"
    dataPath = "../datasets/"
    classLabels = {'ADJ' : 0, 'ADP' : 1, 'ADV' : 2, 'AUX' : 3, 'CCONJ' : 4, 'DET' : 5, 'INTJ' : 6, 'NOUN' : 7, 'NUM' : 8, 'PART' : 9, 'PRON' : 10, 'PROPN' : 11, 'PUNCT' : 12, 'SCONJ' : 13, 'SYM' : 14, 'VERB' : 15, 'X' : 16}
    classes = list(classLabels.keys())
    specialTokens = {'<unknown>', '<start>', '<end>'}
    previousContextSize = 3
    futureContextSize = 3
    epochs = 10
    miniBatchSize = 32
    validationFrequency = 2
    hiddenLayers = [64]
