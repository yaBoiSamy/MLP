import numpy as np


class NeuralNetwork:
    def __init__(self, inputSize, layerSizes):
        self.biases = [np.random.rand(layerSize) for layerSize in layerSizes]
        layerSizes.insert(0, inputSize)
        self.weights = [np.random.rand(layerSizes[i], layerSizes[i - 1]) for i in range(1, len(layerSizes))]

    def forward(self, inputs):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.activations = [np.array(inputs)]
        for weights, biases in zip(self.weights, self.biases):
            self.activations.append(sigmoid(np.dot(weights, self.activations[-1]) + biases))
        return self.activations[-1]

    def backward(self, inputs, tags, growthSpeed=1):
        sigmoid_prime = lambda x: x * (1 - x)
        output = self.forward(inputs)

        Z = [(np.array(tags) - output) * sigmoid_prime(output)]
        for layer in range(len(self.weights) - 1, 0, -1):
            Z.append(np.dot(self.weights[layer].T, Z[-1]) * sigmoid_prime(self.activations[layer]))
        Z = Z[::-1]

        for layer in range(len(Z)):
            self.weights[layer] += growthSpeed * np.outer(Z[layer], self.activations[layer])
            self.biases[layer] += growthSpeed * Z[layer]


speed = 1
XOR = NeuralNetwork(2, [2, 1])

for i in range(10000):
    XOR.backward([0,0], 0, speed)
    XOR.backward([0,1], 1, speed)
    XOR.backward([1,0], 1, speed)
    XOR.backward([1,1], 0, speed)

print("training complete")
while True:
    print(XOR.forward(list(map(int, input().split()))))


