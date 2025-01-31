import numpy as np


class DimensionError(Exception):
    def __init__(self, message):
        super().__init__(message)


def normalize(inputs):
    if not isinstance(inputs, np.ndarray):
        raise TypeError("Input is not a numpy array")
    return (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))


class MLP:
    class Layer:
        def __init__(self, depth, weights, biases):
            weights = np.array(weights) if isinstance(weights, list) else weights
            biases = np.array(biases) if isinstance(biases, list) else biases
            if weights.shape[0] != biases.shape[0]:
                raise DimensionError("Bias shape incompatible with weights shape")
            self.depth = depth
            self.size = biases.shape[0]
            self.inputSize = weights.shape[1]
            self.weights = weights
            self.biases = biases
            self.inputActivations = None

        def forward(self, inputs):  # forward pass uses sigmoid activation function
            if inputs.shape[0] != self.inputSize:
                raise DimensionError(f"Invalid input shape at layer {self.depth}")
            activation = lambda x: 1 / (1 + np.exp(-x))
            self.inputActivations = inputs
            return activation(self.weights @ inputs + self.biases[:, np.newaxis])

        def backward(self, error, learningSpeed):  # backpropagation uses gradient descent, minimizes log loss
            nextError = (self.weights.T @ error) * self.inputActivations * (1 - self.inputActivations)
            self.weights -= (error @ self.inputActivations.T) * learningSpeed
            self.biases -= np.sum(error, axis=1)
            return nextError

        def __str__(self):
            return (f"Depth: {self.depth}\n"
                    f"Input size: {self.inputSize}\n"
                    f"Size:  {self.size}\n"
                    f"Weights: \n{self.weights}\n"
                    f"Biases: \n{self.biases}\n"
                    f"Input activations: \n{self.inputActivations}\n")

    def __init__(self, learningSpeed, weightsArray, biasArray):
        if len(weightsArray) != len(biasArray):
            raise TypeError("Weight and bias arrays of unequal lengths")
        self.learningSpeed = learningSpeed
        self.layers = [self.Layer(depth, weights, biases)
                       for depth, (weights, biases) in enumerate(zip(weightsArray, biasArray))]
        self.output = None
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            if layer1.size != layer2.inputSize:
                raise DimensionError(f"Layers at depths {layer1.depth} and {layer2.depth} have incompatible shapes")

    def forward(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Inputs not a numpy array")
        for layer in self.layers:
            inputs = layer.forward(inputs)
        self.output = inputs
        return self.output

    def backward(self, inputs, tags):
        if not isinstance(tags, np.ndarray):
            raise TypeError("Tags not a numpy array")
        elif self.layers[-1].size != tags.shape[0]:
            raise DimensionError("Tags and output layer have incompatible shapes")
        error = (self.forward(inputs) - tags)/tags.shape[1]
        for layer in self.layers[::-1]:
            error = layer.backward(error, self.learningSpeed)
        return self.evaluate(tags)

    def evaluate(self, tags):  # evaluates the model's most recent forward pass with log loss
        return -np.sum(tags * np.log(self.output) + (1-tags) * np.log(1-self.output))/tags.shape[1]

    def train(self, inputs, tags, epochs):
        # I made the model with column vectors because it is easier for me to understand,
        # as a consequence, the inputs and tags need to be transposed from line vectors to column vectors
        inputs = inputs.T
        tags = tags.T
        performance = []
        for epoch in range(epochs):
            performance.append(self.backward(inputs, tags))
        return performance

    def test(self):
        print("Testing MLP:")
        while True:
            strInput = input(f"Enter all {self.layers[0].inputSize} inputs separated by spaces\n")
            if strInput == "exit":
                break
            inputs = np.array(strInput.split(), dtype=float)[:, np.newaxis]
            print(f"Prediction: \n{self.forward(inputs).round(decimals=3)}\n\n")
        print("Test terminated\n\n")

    @staticmethod
    def RandomNetwork(layerSizes, learningSpeed):
        weightsArray = [np.random.randn(size, inputSize) for inputSize, size in zip(layerSizes[:-1], layerSizes[1:])]
        biasArray = [np.random.randn(size) for size in layerSizes[1:]]
        return MLP(learningSpeed, weightsArray, biasArray)
