import numpy as np


class NeuralNetwork:
    def __init__(self, inputSize, layerSizes):
        self.biases = np.array([np.random.rand(layerSize) for layerSize in layerSizes])
        layerSizes.insert(0, inputSize)
        self.weights = np.array([np.random.rand(layerSizes[i], layerSizes[i - 1]) for i in range(1, len(layerSizes))])

    def forward(self, inputs):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activation = np.array([inputs])
        for weights, biases in zip(self.weights, self.biases):
            np.append(self.activation, sigmoid(np.dot(weights, self.activation[-1]) + biases), axis=0)
        return self.activation[-1]

    def backward(self, inputs, tags):
        sigmoid_prime = lambda x: x * (1 - x)
        output_error = (self.forward(inputs) - np.array(tags))*sigmoid_prime()





