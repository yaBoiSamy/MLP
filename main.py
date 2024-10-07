import numpy as np


class NeuralNetwork:
    def __init__(self, inputSize, layerSizes):
        #np.random.rand(layerSize)
        self.biases = [np.array([0]*layerSize) for layerSize in layerSizes]
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
        z = (output - np.array(tags))*sigmoid_prime(output)
        self.weights += np.outer(z, self.activation[-2])*growthSpeed
        print(self.weights)


speed = 1
XOR = NeuralNetwork(2, [1])
print(XOR.weights)
for i in range(100):
    XOR.backward([0,0], 0, speed)
    XOR.backward([0,1], 1, speed)
    XOR.backward([1,0], 1, speed)
    XOR.backward([1,1], 0, speed)

while True:
    print(XOR.forward(list(map(int, input().split()))))


