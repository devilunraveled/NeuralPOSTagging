class Config :
    ## ==================== DataPaths ====================
    modelSavePath = "checkpoints/"
    plotSavePath = "plots/"
    dataPath = "datasets/"
    
    ## ==================== Split tags ====================
    trainTag = "-train"
    testTag = "-test"
    validationTag = "-dev"
    
    ## ==================== Classes ====================
    classLabels = {'ADJ' : 0, 'ADP' : 1, 'ADV' : 2, 'AUX' : 3, 'CCONJ' : 4, 'DET' : 5, 'INTJ' : 6, 'NOUN' : 7, 'NUM' : 8, 'PART' : 9, 'PRON' : 10, 'PROPN' : 11, 'PUNCT' : 12, 'SCONJ' : 13, 'SYM' : 14, 'VERB' : 15, 'X' : 16}
    classes = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    
    ## ==================== Special Tokens ====================
    specialTokens = {'<unknown>', '<start>', '<end>'}

    ## ========================= Number Of Epochs =========================
    epochs = 10
    
    ## ========================= Mini Batch Size =========================
    miniBatchSize = 16
    
    ## ========================= Validation Frequency =========================
    validationFrequency = 1
    
    ## ========================= MLP Size =========================
    hiddenLayers = [64, 64]
    
    ## ========================= File Extensions =========================
    fileFormat = ".conllu"
    modelFileExtension = ".pt"

class RNNConfig(Config):
    stackSize = 4
    hiddenStateSize = 128

class ANNConfig(Config):
    previousContextSize = 3
    futureContextSize = 3
