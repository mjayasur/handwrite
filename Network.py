'''
A neural network.

:author: Michael Jayasuriya
'''
import math
import random
import numpy

class Network:
    '''
    Represents a 3 layer network of Sigmoid Neurons.
    '''
    def __init__(self, inputSize, outputSize, hiddenSize):
        self.numLayers = 3
        self.layerSizes = [inputSize, outputSize, hiddenSize]
        self.weights = [
            [[random.random() for i in range(inputSize*hiddenSize)]
             for i in range(inputSize)],
            [[random.random() for i in range(hiddenSize*outputSize)]
             for i in range(outputSize)]]
        self.biases = [[0 for n in range(inputSize)],
                       [random.random() for n in range(hiddenSize)],
                       [random.random() for n in range(outputSize)]]
        self.numWeights = 0
        for weights in self.weights:
            self.numWeights += len(weights)

    def propogate(self, inp):
        '''
        propogate the network with input list
        '''
        layers = []
        for i in range(1, 3):
            layerWeights = self.weights[i]
            layerBias = self.biases[i]
            layer = []
            for n in range(self.layerSizes[i]):
                layer.append(0)
                for index in range(self.layerSizes[0]):
                    layer[n] += inp * layerWeights[index][n]
                layer[n] += layerBias[n]
                layer[n] = self.sigmoid(layer[n])
            layers.append(layer)
        return layers[1]

    def train(self, inputs, batch_size):
        '''
        divide inputs into batches of BATCH_SIZE length

        '''
        inputs = random.shuffle(inputs)
        batches = self.partition(inputs, batch_size)
        for batch in batches:
            for input in batch:
                expected = input[1]
                layers = self.propogate(input[0])
                observed = layers[1]
                error = self.cost(expected, observed)




    def backpropogate(self):
        pass

    def cost(self, expected, observed):
        '''
        The least square cost function for this network's current state.
        '''
        return numpy.multiply(numpy.power(
            (numpy.subtract(expected, observed)),2), .5)

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def sigmoid(self, x):
        '''
        returns sigmoid on x
        '''
        return 1/(1 + math.e ** (-x))

    def partition(self, data, size):
        return [data[i : i + size] for i in range(0, len(data), size)]
