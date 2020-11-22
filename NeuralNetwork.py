import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        ''' forward pass '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        ''' backward propagation.
            input: dvalues  the gradient of the forward layer of neurons
        '''
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # create the gradient for this layer
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    ''' Rectified Linear Unit Activation Function '''
    def __init__(self):
        pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # ReLU function f(x) = {0, x <=0, x, x > 0}
        # so its derivative is 0 if the input was zero.
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        ''' calcuate normalized probabilities in the forward pass '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def backward(self, dvalues):
        ''' create the gradient vector for backpropagation '''
        self.dinputs = np.empty_like(dvalues)

        for i, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        no_of_samples = len(dvalues)
        # turn y_true into discrete values if y_true is one-hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(no_of_samples), y_true] -= 1
        self.dinputs = self.dinputs / no_of_samples

class Loss:
    ''' Generic loss class '''
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def calculate(self, output, y):
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
        # I'm not sure if the latter is entirely necessary since only
        # 0 is clipped, and by a very small value.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # categorical values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(no_of_samples), y_true]
        # one-hot encoded values
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # return the loss as the negative log likelihoods
        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        no_of_samples = len(dvalues)
        no_of_labels = len(dvalues[0])
        # transform sparse data to one-hot vectors
        if len(y_true.shape) == 1:
            y_true = np.eye(no_of_labels)[y_true]
        # calculate & normalize gradient
        self.dinputs = (-y_true / dvalues) / no_of_samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate /\
                                         (1. + self.decay * self.iterations)
                                         
    def update_params(self, layer):
        if self.momentum:
            # create the initial momentums
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            # determine the new weight & bias updates by preserving some
            # of the previous weight & bias updates through the use of momentum
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # weight & bias updates using only current gradient
            weight_updates += -self.current_learning_rate * layer.dweights
            bias_updates += -self.current_learning_rate * layer.dbiases

        ''' Updates the parameters for a layer of the network '''
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate /\
                                         (1. + self.decay * self.iterations)
                                         
    def update_params(self, layer):
        ''' Update parameters using adaptive gradient method. A cache of the
            weight and bias gradients is kept as a sum of squares of previous
            gradients.
        '''
        # Initialize caches
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Track sum of square of all gradients used to update model parameters
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Update model parameters normalize to the square root of the
        # sum of square of all gradients used (cache)
        layer.weights += -self.current_learning_rate * layer.dweights / \
                          (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        ''' Keep track of number of epochs. This variable is used to update
            the variable learning rate. This method should be called after
            any parameter update.
        '''
        self.iterations += 1

class Optimizer_RMSprop:
    ''' Optimizes a neural network using Root Mean Square PROPagation '''
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1. + self.decay * self.iterations)

    def update_params(self, layer):
        # Initialize weight and bias caches
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases ** 2
        
        # parameter update normalized using RMS
        layer.weights += -self.current_learning_rate * layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
                         (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1    

if __name__ == '__main__':
    X, y = spiral_data(100, 3)

    dense1 = Layer_Dense(2, 64)
    dense2 = Layer_Dense(64, 3)
    activation1 = Activation_ReLU()
    loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
    # optimizer = Optimizer_SGD(learning_rate=0.98, decay=1.0e-3, momentum=0.9)
    # optimizer = Optimizer_Adagrad(learning_rate=0.9, decay=1.0e-4)
    optimizer = Optimizer_RMSprop(decay=1e-4)

    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        if not epoch % 250:
            print(f'epoch: {epoch}',
                  f'\tacc: {accuracy:.3f}',
                  f'\tloss: {loss:.3f}'
                  f'\tlr: {optimizer.current_learning_rate:0.6f}')

        #backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()