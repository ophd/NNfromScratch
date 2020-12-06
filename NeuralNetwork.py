import numpy as np
import nnfs
from nnfs.datasets import sine_data

nnfs.init()

class Layer_Input:
    def forward(self, inputs):
        self.output = inputs

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # init weights & biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # set strength of regularization
        # lambda values for L1 & L2 regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

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

        # Gradients on regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights <0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # create the gradient for this layer
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Dropout:
    def __init__(self, rate):
        ''' input is dropout rate, but it is saved in the layer
            as the retention rate
        '''
        self.rate = 1 - rate
    
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) \
                           / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


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
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


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

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate(self, output, y):
        ''' Calculates the data and regularization losses
            given model output and ground-truth values '''
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss, self.regularization_loss()
    
    def regularization_loss(self, layer):
        ''' This function calculates the loss from L1 & L2 regularization
            on the weights & biases
        '''
        regularization_loss = 0
        for layer in self.trainable_layers:
            # L1 for weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))
            # L2 for weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights * layer.weights)
            # L1 for biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))
            # L2 for biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases * layer.biases)
        return regularization_loss

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


class Loss_BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # clip data to avoid division by zero
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clip) +
                          (1 - y_true) * np.log(1 - y_pred_clip))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        no_of_samples = len(dvalues)
        no_of_outputs = len(dvalues[0])

        # clip data to avoid division by zero
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate the gradient
        self.dinputs = -(y_true / clipped_dvalues -
                        (1 - y_true) / (1 - clipped_dvalues)) / no_of_outputs
        # Normalize the gradient
        self.dinputs = self.dinputs / no_of_samples

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        no_of_samples = len(dvalues)
        no_of_outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / no_of_outputs
        self.dinputs = self.dinputs / no_of_samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        no_of_samples = len(dvalues)
        no_of_outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / no_of_outputs
        self.dinputs = self.dinputs / no_of_samples

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

class Optimizer_Adam:
    ''' An optimizer that uses Adaptive Momentum to train a neural network
    '''
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1. + self.decay * self.iterations)
    
    def update_params(self, layer):
        # initializes momentums and caches for weight and bias updates
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradient
        layer.weight_momentums = self.beta1 * layer.weight_momentums + \
            (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + \
            (1 - self.beta1) * layer.dbiases
        # Calculate bias-corrected momentums
        # iterations + 1 to void division by zero
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta1 ** (self.iterations + 1))

        # update weight and bias caches with square of current gradient
        layer.weight_cache = self.beta2 * layer.weight_cache + \
            (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + \
            (1 - self.beta2) * layer.dbiases ** 2
        # Calculate bias-corrected gradient caches
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta2 ** (self.iterations + 1))
        
        # update layer weights and biases using SDG, with gradients
        # corrected using momentum and root mean square of gradients
        layer.weights += -self.current_learning_rate * \
                            weight_momentums_corrected / \
                            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                            bias_momentums_corrected / \
                            (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Model:
    ''' generic neural network class

        This class contains the building blocks of a neural network model,
        including the layers, optimizer, loss function, and training.
    '''
    def __init__(self):
        self.layers = []

    def add(self, layer):
        ''' adds a layer to the neural network model '''
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def train(self, X, y, *, epochs=1, print_every=1):
        for epoch in range(1, epochs+1):
            output = self.forward(X)
            print(output)
            exit()
    
    def finalize(self):
        self.input_layer = Layer_Input()
        self.trainable_layers = []

        no_of_layers = len(self.layers)

        for i in range(no_of_layers):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < no_of_layers - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
            
            # Gather list of trainable layers
            # Trainable layers have weights and biases
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

    def forward(self, X):
        ''' Performs a forward pass through the neural network '''
        self.input_layer.forward(X)
        
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output

if __name__ == '__main__':
    X, y = sine_data()
    
    model = Model()

    model.add(Layer_Dense(1, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 1))
    model.add(Activation_Linear())

    model.set(
        loss=Loss_MeanSquaredError(),
        optimizer=Optimizer_Adam(learning_rate=0.005, decay=1.e-3)
    )

    model.finalize()
    model.train(X, y, epochs=10000, print_every=100)