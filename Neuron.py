'''
File for neurons.

:author: Michael Jayasuriya
'''

class Neuron:
    '''
    Represents a neuron with an input function.
    '''
    def __init__(self):
        self.pathways = {}
        self.activation = 0


    def input(self, inp):
        '''
        Input at a dendrite.
        :return: true if successful
        '''
        self.activation += inp

    def connect(self, neuron, weight):
        self.pathways[neuron] = weight

    def bias(self, b):
        self.b = b