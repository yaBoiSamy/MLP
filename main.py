import numpy as np


class NeuralNetwork:
    def __init__(self, inputSize, layerSizes):
        self.biases = [np.random.rand(layerSize) for layerSize in layerSizes]
        layerSizes.insert(0, inputSize)
        self.weights = [np.random.rand(layerSizes[i], layerSizes[i - 1]) for i in range(1, len(layerSizes))]

    def forward(self, inputs):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activation = [np.array(inputs)]
        for weights, biases in zip(self.weights, self.biases):
            self.activation.append(sigmoid(np.dot(weights, self.activation[-1]) + biases))
        return self.activation[-1]

    def backward(self, inputs, tags, growthSpeed=1):
        sigmoid_prime = lambda x: x * (1 - x)
        output = self.forward(inputs)
        Z = [(np.array(tags) - output) * sigmoid_prime(output)]
        for layer, weights in enumerate(self.weights[:1:-1], 2):
            Z.append(np.dot(weights.T, Z[-1])*sigmoid_prime(self.activation[-layer]))
        Z = Z[::-1]
        for layer in range(len(Z)):
            self.weights[layer] += np.outer(Z[layer], self.activation[layer])*growthSpeed
            self.biases[layer] += Z[layer]*growthSpeed



speed = 1
XOR = NeuralNetwork(2, [2, 1])
print(XOR.weights)
for i in range(100):
    XOR.backward([0,0], 0, speed)
    XOR.backward([0,1], 1, speed)
    XOR.backward([1,0], 1, speed)
    XOR.backward([1,1], 0, speed)
print(XOR.weights)
while True:
    print(XOR.forward(list(map(int, input().split()))))


