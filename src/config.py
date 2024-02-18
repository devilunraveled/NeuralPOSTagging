class Config :
    modelSavePath = "../checkpoints/"
    dataPath = "../datasets/"
    classLabels = {'ADJ' : 0, 'ADP' : 1, 'ADV' : 2, 'AUX' : 3, 'CCONJ' : 4, 'DET' : 5, 'INTJ' : 6, 'NOUN' : 7, 'NUM' : 8, 'PART' : 9, 'PRON' : 10, 'PROPN' : 11, 'PUNCT' : 12, 'SCONJ' : 13, 'SYM' : 14, 'VERB' : 15, 'X' : 16}
    specialTokens = {'<start>', '<end>', '<unknown>'}
    previousContextSize = 3
    futureContextSize = 3
    epochs = 4
    miniBatchSize = 64
