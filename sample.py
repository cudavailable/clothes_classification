class Sample:
    def __init__(self, img=None, classPair=None, labelList=None, feat=None) -> None:
        # self.path = path
        self.img = img
        self.classColor = classPair[0]
        self.classCos = classPair[1]

        self.labelColor = labelList[:6]
        self.labelCos = labelList[6:]
        self.feat = feat
        self.pred = None