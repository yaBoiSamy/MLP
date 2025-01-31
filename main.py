import numpy as np
import matplotlib.pyplot as plt
import MLP_class as ml

trainingInputs = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
)

trainingTags = np.array(
    [
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0]
    ]
)

myNetwork = ml.MLP.RandomNetwork([2, 3, 2], 0.1)
performance = myNetwork.train(trainingInputs, trainingTags, 30000)
plt.plot(performance)
plt.show()
myNetwork.test()