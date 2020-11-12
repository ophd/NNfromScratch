import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    ''' Rectified Linear Unit Activation Function '''
    def __init__(self):
        pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        ''' calcuate normalized probabilities in the forward pass '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    ''' Common loss class '''
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def calcuate(self, output, y):
        ''' Calculates the data and regularization losses
            given model output and ground-truth values '''
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    ''' cross-entropy loss '''
    def forward(self, y_pred, y_true):
        no_of_samples = len(y_pred)
        # clipping eliminates 0 and 1 values.
        # clip 0 value to avoid log(0) errors
        # clip 1 to even out the mean from clipping 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # categorical values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(no_of_samples), y_true]
        # one-hot encoded values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # return the loss as the negative log likelihoods
        return -np.log(correct_confidences)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]
]

X, y = spiral_data(100, 3)

layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
layer2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
layer2.forward(activation1.output)
activation2.forward(layer2.output)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print('acc:', accuracy)

