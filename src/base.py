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
        text = f"{Color.YELLOW}Sentence ID : {self.id}{Color.END}\n"
        for token,POStag in zip(self.original.split(" "), self.POStags):
            text += f"{Color.CYAN}{token:<12.10} {Color.BOLD}{Color.GREEN}{POStag:<8.4}{Color.END} \n"

        return text

class DataPoint :
    def __init__(self, dataPoint):
        self.dataPoint = [token[0] for token in dataPoint]

    def __str__(self):
        return str(self.dataPoint)


def OneHotEncoding(totalSize, index) :
    oneHotEncoding = [0] * totalSize
    oneHotEncoding[index] = 1
    return oneHotEncoding
