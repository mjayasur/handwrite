import numpy as np
import struct
import csv
import matplotlib.pyplot as plt


class MnistRead:
    def __init__(self, trainImages, trainLables, testImages, testLabels):
        '''
        :param trainImages: filename
        :param trainLables: filename
        :param testImages: filename
        :param testLabels: filename
        '''
        self.trainInFile = trainImages
        self.trainOutFile = trainLables
        self.testInFile = testImages
        self.testOutFile = testLabels
        self.process()

    def process(self):
        with open(self.trainInFile, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))

        self.trainIn = data

        with open(self.testInFile, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))
        self.testIn = data
        with open(self.trainOutFile, 'rb') as f:
            f.read(8)

            data = list(f.read())
        self.trainOut = data
        with open(self.testOutFile, 'rb') as f:
            f.read(8)

            data = list(f.read())
        self.testOut = data
        # trainInputsWriter = csv.writer(open("mnist/train/images.csv", 'w'))
        self.trainingDataIn = []
        for inp in self.trainIn:
            whole = []
            for row in inp:
                whole.extend(row)
            self.trainingDataIn.append(whole)
            # trainInputsWriter.writerow(whole)
        # testInputsWriter = csv.writer(open("mnist/test/images.csv", 'w'))
        self.testDataIn = []
        for inp in self.testIn:
            whole = []
            for row in inp:
                whole.extend(row)
            self.testDataIn.append(whole)
            # testInputsWriter.writerow(whole)
        self.trainingDataOut = []
        for out in self.trainOut:
            self.trainingDataOut.append([0 for i in range(out)] + [1] + [0 for i in range(9 - out)])
        self.testingDataOut = []
        for out in self.testOut:
            self.testingDataOut.append([0 for i in range(out)] + [1] + [0 for i in range(9 - out)])
        self.train = zip(self.trainingDataIn, self.trainingDataOut)
        self.test = zip(self.testDataIn, self.testingDataOut)






