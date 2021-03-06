import numpy as np
import nnfs
import pickle
import copy
from analyse_fashion_MNIST import create_data_mnist
from nnfs.datasets import sine_data, spiral_data

nnfs.init()

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # init weights & biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # set strength of regularization
        # lambda values for L1 & L2 regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def get_parameters(self):
        ''' retrieves the parameters of the dense layer. '''
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        ''' sets the parameters of the dense layers. '''
        self.weights = weights
        self.biases = biases

    def forward(self, inputs, training):
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
    
    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) \
                           / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:
    ''' Rectified Linear Unit Activation Function '''
    def __init__(self):
        pass

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # ReLU function f(x) = {0, x <=0, x, x > 0}
        # so its derivative is 0 if the input was zero.
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        ''' calcuate normalized probabilities in the forward pass '''
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues, training):
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
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    # def __init__(self):
    #     self.activation = Activation_Softmax()
    #     self.loss = Loss_CategoricalCrossEntropy()

    # def forward(self, inputs, y_true):
    #     self.activation.forward(inputs)
    #     self.output = self.activation.output
    #     return self.loss.calculate(self.output, y_true)
        
    def backward(self, dvalues, y_true):
        no_of_samples = len(dvalues)
        # turn y_true into discrete values if y_true is one-hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(no_of_samples), y_true] -= 1
        self.dinputs = self.dinputs / no_of_samples


class Accuracy:
    ''' Base class for prediction accuracy '''
    def compare(self):
        raise NotImplementedError

    def calculate(self, predictions, y):
        ''' calculate prediction accuracy given predictions and ground-truth
        values '''
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_cout += len(comparisons)

        return accuracy
    

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_cout

        return accuracy
        

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_cout = 0


class Accuracy_Regression(Accuracy):
    ''' Accuracy calculations for a regression model '''
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    ''' Accuracy for a classifical model '''
    def __init__(self, *, binary=False):
        self.binary = binary
    
    def init(self, y):
        ''' this function is not needed for this model type but
            needs to exist to avoid throwing and exception
        '''
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Loss:
    ''' Generic loss class '''
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate(self, output, y, *, include_regularization=False):
        ''' Calculates the data and regularization losses
            given model output and ground-truth values '''
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
    def regularization_loss(self):
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
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

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
        self.softmax_classifier_output = None
    
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy
    
    def get_parameters(self):
        ''' Retrieves parameters for all deep layers of the model '''
        parameters = [layer.get_parameters() for layer in self.trainable_layers]
        return parameters
    
    def set_parameters(self, parameters):
        ''' Sets the parameters for all deep layers of the model. '''
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def save_parameters(self, path):
        ''' Saves model parameters to file using pickling '''
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        ''' Loads model parameters from file '''
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    
    def save(self, path):
        ''' Saves a copy of the model. '''
        model = copy.deepcopy(self)
        # reset accumulated values from loss & accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        # reset input layer & gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        # reset inputs, outputs, and gradients from all layers
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        # Save model to file
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(path):
        ''' Loads a saved model from file. '''
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, 
              validation_data=None):
        self.accuracy.init(y)

        # training step within an epoch
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        
        # break down training data in batches
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # catch any data missed by the integer division
            if train_steps * batch_size < len(X):
                train_steps += 1
            
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                # catch any data missed by the integer division
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')

            # if data is processed in batches, this resets the cumulative
            # tracking of batches for the new epoch
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
            
                # forward pass
                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y, 
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # backward pass
                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)            
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'\tstep: {epoch}',
                        f'acc: {accuracy:.3f}',
                        f'loss: {loss:.3f} (',
                        f'data loss: {data_loss:.3f},',
                        f'reg loss: {regularization_loss:.3f})',
                        f'lr: {self.optimizer.current_learning_rate:.5f}'
                    )
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training')
            print(f'acc: {epoch_accuracy:.3f}',
                  f'loss: {epoch_loss:.3f} (',
                  f'data_loss: {epoch_data_loss:.3f},',
                  f'reg_loss: {epoch_regularization_loss:.3f})',
                  f'lr: {self.optimizer.current_learning_rate:.5f}')

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
            
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation: acc: {validation_accuracy:.3f}',
              f'loss: {validation_loss:.3f}')
    
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
                self.output_layer_activation = self.layers[i]
            
            # Gather list of trainable layers
            # Trainable layers have weights and biases
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # if output activation is Softmax and the loss function is 
        # Categorical Cross-Entropy, we can use an optimized method
        # to calculate the gradient
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossEntropy):
           self.softmax_classifier_output = \
               Activation_Softmax_Loss_CategoricalCrossEntropy()

    def forward(self, X, training):
        ''' Performs a forward pass through the neural network '''
        self.input_layer.forward(X, training)
        
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            # Using the combined Softmax activation and cross-entropy loss
            # for efficiency, the backward method of the last layer (softmax
            # activation) is not used and dinputs must be set manually for
            # the last activation layer.
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

            
    def predict(self, X, *, batch_size=None):
        ''' Given an array of inputs, this method predicts an
            output using the trained model.
        '''
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
        return np.vstack(output)

if __name__ == '__main__':
    fashion_mnist_labels = {
        0: 'T-shirt/top',
        1: 'Trousers',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X, y = X[keys], y[keys]

    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    model = Model.load('fashion_mnist.model')
    confidences = model.predict(X_test[:5])
    predictions = model.output_layer_activation.predictions(confidences)
    
    for prediction in predictions:
        print(fashion_mnist_labels[prediction])
    # model.evaluate(X_test, y_test)
    # model = Model()

    # model.add(Layer_Dense(X.shape[1], 128))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(128, 128))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(128, 10))
    # model.add(Activation_Softmax())

    # model.set(
    #     loss=Loss_CategoricalCrossEntropy(),
    #     optimizer=Optimizer_Adam(decay=1e-3),
    #     accuracy=Accuracy_Categorical()
    # )

    # model.finalize()

    # model.train(X, y, validation_data=(X_test, y_test),
    #             epochs=10, batch_size=128, print_every=100)

    # model.evaluate(X_test, y_test)

    # parameters = model.get_parameters()

    # # New Model
    # model = Model()

    # model.add(Layer_Dense(X.shape[1], 128))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(128, 128))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(128, 10))
    # model.add(Activation_Softmax())

    # model.set(
    #     loss=Loss_CategoricalCrossEntropy(),
    #     accuracy=Accuracy_Categorical()
    # )

    # model.finalize()
    # model.set_parameters(parameters)
    # model.evaluate(X_test, y_test)

    # model.save('fashion_mnist.model')