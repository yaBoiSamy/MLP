import numpy as np


class NeuralNetwork:
    def __init__(self, inputSize, layerSizes):
        self.biases = np.array([np.random.rand(layerSize) for layerSize in layerSizes])
        layerSizes.insert(0, inputSize)
        self.weights = np.array([np.random.rand(layerSizes[i], layerSizes[i - 1]) for i in range(1, len(layerSizes))])

    def forward(self, inputs):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        currentLayer = inputs
        for weights, biases in zip(self.weights, self.biases):
            currentLayer = sigmoid(np.dot(weights, currentLayer) + biases)
        return currentLayer

    def backward(self):
        sigmoid_prime = lambda x: x * (1 - x)

