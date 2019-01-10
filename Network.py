'''
A neural network.

:author: Michael Jayasuriya
'''
from random import random, shuffle
import numpy

class Network:
    '''
    Represents a layerSize network of Sigmoid Neurons.
    '''
    def __init__(self, layerSizes, learningRate = 3.0):
        '''
        self.weights is formatted as a list of layers, with each layer containing lists of random ints.
        :param layerSizes determines number of neurons in each layer
        :param learningRate: for SGD
        '''
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.weights = []
        self.biases = [numpy.random.randn(y, 1).flatten() for y in layerSizes[1:]]
        self.weights = [numpy.random.randn(y, x)
                        for x, y in zip(layerSizes[:-1], layerSizes[1:])]
        self.learningRate = learningRate

    def propogate(self, inp):
        '''
        propogate the network with input list
        '''
        assert len(inp) == self.layerSizes[0], "incorrect input size of " + str(len(inp))
        layers = [inp]
        for bias, weight in zip(self.biases, self.weights):
            layers.append(self.sigmoid(numpy.dot(weight, layers[-1]) + bias))
        return layers
    def train(self, inp, batch_size=100, epochs=10):
        self.gradient_descent(inp, batch_size, epochs)

    def gradient_descent(self, inputs, batch_size, epochs):
        '''
        divide inputs into batches of BATCH_SIZE length and perform GD

        '''
        shuffle(inputs)
        batches = self.partition(inputs, batch_size)
        for batch in batches:
            for f in range(epochs):
                for inp in batch:
                    expected = inp[1]
                    gradients = self.backpropogate(inp[0], expected)
                    biasGradient = gradients[0]
                    self.biases = numpy.subtract(self.biases, numpy.multiply(biasGradient, self.learningRate))
                    weightsGradient = gradients[1]
                    for layer in range(len(self.weights)):
                        for neuron in range(len(self.weights[layer])):
                            self.weights[layer][neuron] = numpy.subtract(self.weights[layer][neuron], numpy.multiply(weightsGradient[layer][neuron], self.learningRate))

    def observed(self, inp):
        output = self.propogate(inp)[2]
        maximum = 0
        for i in range(len(output)):
            if output[i] > output[maximum]:
                maximum = i
        return maximum



    def backpropogate(self, inp, expected):
        '''
        adjust network's weights by SGD
        :param input
        :param expected is the target values
        '''
        biasgradient = [numpy.zeros(b.shape) for b in self.biases]
        weightgradient = [numpy.zeros(w.shape) for w in self.weights]
        layers = [inp]
        act = inp
        zlayers = []
        for bias, weight in zip(self.biases, self.weights):
            aZ = numpy.dot(weight, act)
            aZ += bias
            zlayers.append(aZ)
            act = self.sigmoid(aZ)
            layers.append(act)


        deltacost = self.dcost(layers[-1], expected)
        sigmoidprime = self.dsigmoid(zlayers[-1])
        lastdeltaerror = numpy.multiply(deltacost, sigmoidprime)
        biasgradient[-1] = lastdeltaerror

        weightgradient[-1] = numpy.dot(lastdeltaerror.reshape(len(lastdeltaerror), 1), layers[-2].reshape(1, self.layerSizes[-2]))




        for i in range(2, len(layers) - 1):
            prevLayer = numpy.array(layers[-i-1]).reshape(1, self.layerSizes[-i-1])
            currWeights = self.weights[-i+1]
            lastdeltaerror = numpy.dot(currWeights.transpose(), lastdeltaerror) * self.dsigmoid(zlayers[-i])
            biasgradient[-i] = lastdeltaerror

            weightgradient[-i] = numpy.dot(lastdeltaerror, prevLayer.transpose())
        return [biasgradient, weightgradient]

    def cost(self, observed, expected):
        '''
        The least square cost function for this network's current state.
        '''
        return (observed - expected)
    def dcost(self, observed, expected):
        if (observed.shape == (10,30)):
            return True
        return numpy.subtract(observed, expected)

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def sigmoid(self, x):
        return 1.0 / (1.0 + numpy.exp(-x))

    def partition(self, data, size):
        return [data[i : i + size] for i in range(0, len(data), size)]

