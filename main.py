import numpy as np
import random

class NeuralNetwork:
    def __init__(self, inputSize, layerSizes, growthSpeed):
        self.biases = [np.random.rand(layerSize) for layerSize in layerSizes]
        layerSizes.insert(0, inputSize)
        self.weights = [np.random.rand(layerSizes[i], layerSizes[i - 1]) for i in range(1, len(layerSizes))]
        self.growthSpeed = growthSpeed
        self.layers = layerSizes

    def smushFunc(self, x):
        return np.tanh(x)
        # 1 / (1 + np.exp(-x))

    def smushFunc_prime(self, x):
        return 1 - np.tanh(x) ** 2
        # x * (1 - x)

    def forward(self, inputs):
        self.activations = [np.array(inputs)]
        for weights, biases in zip(self.weights, self.biases):
            self.activations.append(self.smushFunc(np.dot(weights, self.activations[-1]) + biases))
        return self.activations[-1]

    def backward(self, inputs, tags):
        output = self.forward(inputs)

        Z = [np.array(tags) - output]
        for layer in range(len(self.weights) - 1, 0, -1):
            Z.append(np.dot(self.weights[layer].T, Z[-1]) * self.smushFunc_prime(self.activations[layer]))
        Z = Z[::-1]

        for layer in range(len(Z)):
            self.weights[layer] += self.growthSpeed * np.outer(Z[layer], self.activations[layer])
            self.biases[layer] += self.growthSpeed * Z[layer]

    def answerQuality(self, inputs, expected):
        outputs = self.forward(inputs)
        expected = np.array(expected)
        return abs((outputs-expected)/expected)*100

def train(NNs, inputs, expected):
    for inputt, answer in zip(inputs, expected): # Train the networks
        for NN in NNs:
            NN.backward(inputt, answer)
    print("training complete")

def evaluate(NNs, inputs, expected):
    scorePrefixes = [0 for NN in NNs]
    for inputt, answer in zip(inputs, expected):
        for i, NN in enumerate(NNs):
            scorePrefixes[i] += sum(NN.answerQuality(inputt, answer))/len(answer)
    return {scorePrefix/len(inputs):NN for scorePrefix, NN in zip(scorePrefixes, NNs)}


multipliers = [NeuralNetwork(2, [random.randint(1, 4) for i in range(random.randint(1, 3))] + [1], random.uniform(0.05, 0.005)) for i in range(100)] # Initialize MLPs
DATASET = [[random.uniform(0.1, 1), random.uniform(0.1, 1)] for i in range(1000)] # Create the training inputs
TAGS = [[data[0]*data[1]] for data in DATASET] # Create the training answers
train(multipliers, DATASET, TAGS)

TESTCASES = [[random.uniform(0.1, 1), random.uniform(0.1, 1)] for i in range(100)] # Create the testcases
TESTCASE_ANSWERS = [[inputt[0]*inputt[1]] for inputt in TESTCASES] # Generate the answers to these tescases
multipliers = evaluate(multipliers, TESTCASES, TESTCASE_ANSWERS) # Evaluate general performance


bestScores = list(map(float, sorted(multipliers.keys())))
print("best scores:", bestScores[:10])
print("worst scores:", bestScores[-10:])

bestAI = multipliers[bestScores[0]]
print("\nbest MLP stats:")
print("  structure -", bestAI.layers)
print("  growth speed - ", bestAI.growthSpeed, "\n")

while True: # Test the best MLP
    inputs = list(map(float, input("Enter AI inputs: ").split()))
    networkResults = bestAI.forward(inputs)
    print("  network results -", networkResults)
    print("  answer score -", bestAI.answerQuality(inputs, [inputs[0]*inputs[1]]), "\n")