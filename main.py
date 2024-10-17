import numpy as np
import random

"""
                                            --  HOW MLP's WORK  --

-- Variable definitions --

Say we have an input layer at depth 0, hidden layers at depths 1 to L-1, and an output layer at depth L. 
This hypothetical MLP does not use batch training. Let's define the following variables:
A(n): The vector with the activation values for each neuron at layer n
W(n): The matrix with the weights that provide the activations for layer n of the network ; W(n)_ij connects the A(n)_i neuron to the A(n-1)_j neuron
B(n): The vector with the biases that provide the activations for layer n of the network
Z(n): The vector corresponding to the result of W(L)A(L-1) + B(L)
C: The loss vector, calculated using mean squared error between the output layer and tags
F(x): The activation function
F'(x): The derivative of the activation function


-- Forward propagation --

The inputs are fed as activations of the A(0) layer
To calculate the activation of the following layer A(1), or in general any A(n) from A(n-1), we do:
Z(n) = W(n)A(n-1) + B(n)
A(n) = F(Z(N))

The matrices W and the Biases B serve as a form of parameters that we can tweak to adjust the behavior of the MLP
The MLP is basically a huge function that can adjust itself by tweaking its own weights and biases
We can calculate the accuracy, aka the Cost (C) of the MLP using the mean squared error between the outputs (O) and expected results (Y): C(W, B) = (Y - O(W, B))^2


-- Backward propagation --

The goal of backprop is to train the AI using datasets with answers attached to inputs. 
To do this training, we calculate the gradient of the weights and biases in relation to the cost function (∇C(W, B)).
This gradient, if added to the weights and biases themselves, allows for gradient descent, which converges toward the local minima of the cost function
Our objective is therefore to calculate dC/dW and dC/dB
Note that everytime I use the dY/dX notation, I am referring to a partial derivative, not a traditional one.

The Error term E(n) is a substep to calculating dC/dW or dC/dB, it corresponds to dC/dZ(n)
To calculate the error term E(L), we do:
E(L)_i = dC_i/dA(L)_i * dA(L)_i/dZ(L)_i = dC_i/dA(L)_i *  F'(Z(L)_i)
This can be generalized to a vector operation:
E(L) = (C-A(L))F'(Z(L))
This should give us a vector of length len(A(L)).

Now that we have the error term for the output layer, we calculate it recursively for all other layers
To calculate the next error term E(L-1) from E(L), or any error term E(n-1) from E(n), we do:
E(n-1)_i = ∑_j E(n)_j * W(n)_ji
This can be generalized via matrix-vector product:
E(n-1) = (W(n)^T)E(n)
This should give us a vector of length len(A(n-1))

To calculate the gradient of the cost for each of the weights Ew(n), we do:
Ew(n)_ij = E(n)_i * dZ(n)_i/dW(n)_ij = E(n)_i * A(n-1)_j
This can be generalized to an outer product:
Ew(n) = E(n)(A(n-1)^T)
This should give us a matrix of dimensions len(A(n))xlen(A(n-1))

To calculate the gradient of the cost for each of the biases Eb(n), we do:
Eb(n)_i = E(n)_i * dZ(n)/dB(n) = E(n)_i
This can be generalized to:
Eb(n) = E(n)
This should give us a vector of length len(A(n))

Now that we know both dC/dW and dC/dB, we can update their previous values:
W(n) -= Ew(n) * growth_factor
B(n) -= Eb(n) * growth_factor

"""


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

# NETWORK PARAMETERS
HIDDEN_LAYER_COUNT = 3
NODES_PER_HIDDEN_LAYER = 4
INPUT_LAYER_SIZE = 1
OUTPUT_LAYER_SIZE = 1
GROWTH_SPEED_BOUNDS = (0.05, 0.5)
MLP_COUNT = 100
answerFunc = lambda x: np.sin(x[0])

networks = [NeuralNetwork(INPUT_LAYER_SIZE, [random.randint(1, NODES_PER_HIDDEN_LAYER) for i in range(random.randint(1, HIDDEN_LAYER_COUNT))] + [OUTPUT_LAYER_SIZE], random.uniform(GROWTH_SPEED_BOUNDS[0], GROWTH_SPEED_BOUNDS[1])) for i in range(MLP_COUNT)] # Initialize MLPs


# TRAINING
DATASET = [[random.uniform(0.1, 1)] for i in range(10000)] # Create the training inputs
TAGS = [[answerFunc(data)] for data in DATASET] # Create the training answers
train(networks, DATASET, TAGS)


# EVALUATION
TESTCASES = [[random.uniform(0.1, 1)] for i in range(100)] # Create the testcases
TESTCASE_ANSWERS = [[answerFunc(inputt)] for inputt in TESTCASES] # Generate the answers to these tescases
networks = evaluate(networks, TESTCASES, TESTCASE_ANSWERS) # Evaluate general performance


bestScores = list(map(float, sorted(networks.keys())))
print("best scores:", bestScores[:10])
print("worst scores:", bestScores[-10:])

bestAI = networks[bestScores[0]]
print("\nbest MLP stats:")
print("  structure -", bestAI.layers)
print("  growth speed - ", bestAI.growthSpeed, "\n")

while True: # Test the best MLP
    inputs = list(map(float, input("Enter AI inputs: ").split()))
    networkResults = bestAI.forward(inputs)
    print("  network results -", networkResults)
    print("  answer score -", bestAI.answerQuality(inputs, [answerFunc(inputs)]), "\n")
