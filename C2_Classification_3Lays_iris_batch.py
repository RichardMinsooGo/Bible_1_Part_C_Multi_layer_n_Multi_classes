#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

"""
Optimizers
"""

import numpy as np

class Optimizer:
    def __init__(self, learning_rate=None, name=None):
        self.learning_rate = learning_rate
        self.name = name

    def config(self, layers):
        # sets up empty cache dictionaries 
        pass

    def optimize(self, idx, layers: list, grads: dict, *args):
        '''# Args: Takes in idx of the layer, list of the layers and the gradients as a dictionary 
            Performs updates in the list of layers passed into it'''
        pass 

class SGDM(Optimizer):
    '''  Momentum builds up velocity in any direction that has consistent gradient'''

    def __init__(self, learning_rate=1e-2, mu_init=0.5, max_mu=0.99, demon=False, beta_init=0.9, **kwargs):
            super().__init__(**kwargs)
            self.mu_init = mu_init
            self.max_mu = max_mu
            self.demon = demon
            if self.demon:
                self.beta_init = beta_init
            self.m = dict()

    def config(self, layers):
        for i in layers.keys():
            self.m[f'W{i}'] = 0
            self.m[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps):
        # increase mu by a factor of 1.2 every epoch until max_mu is reached (only applicable for momentum and nesterov momentum)
        mu = min(self.mu_init * 1.2 ** (epoch_num - 1), self.max_mu)

        if self.demon:
            p_t = 1 - epoch_num / self.epochs 
            mu = self.beta_init * p_t / ((1 - self.beta_init) + self.beta_init * p_t) 

        self.m[f'W{idx}'] = self.m[f'W{idx}'] * mu - self.learning_rate * grads[f'dW{idx}']
        self.m[f'b{idx}'] = self.m[f'b{idx}'] * mu - self.learning_rate * grads[f'db{idx}']

        layers[idx].W += self.m[f'W{idx}']
        layers[idx].b += self.m[f'b{idx}']

class Adam(Optimizer):
    '''One of the most popular first-order gradient descent algorithms with momentum estimate
        terms : https://arxiv.org/pdf/1412.6980.pdf'''
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                 weight_decay=False, gamma_init=1e-5, decay_rate=0.8, demon=False, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2 
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        if self.weight_decay:
            self.gamma_init = gamma_init
            self.decay_rate = decay_rate
        self.demon = demon
        self.m = dict()  # first moment estimate 
        self.v = dict()  # second raw moment estimate 

    def config(self, layers):
        for i in layers.keys():
            self.m[f'W{i}'] = 0
            self.m[f'b{i}'] = 0
            self.v[f'W{i}'] = 0
            self.v[f'b{i}'] = 0

    def optimize(self, idx, layers, grads, epoch_num, steps): 
        dW = grads[f'dW{idx}']
        db = grads[f'db{idx}']
        if self.demon:
            p_t = 1 - epoch_num / self.epochs
            beta1 = self.beta1 * (p_t / (1 - self.beta1 + self.beta1 * p_t))
        else:
            beta1 = self.beta1

        # weights
        self.m[f'W{idx}'] = beta1 * self.m[f'W{idx}'] + (1 - beta1) * dW
        self.v[f'W{idx}'] = self.beta2 * self.v[f'W{idx}'] + (1 - self.beta2) * dW ** 2 
        
        # biases
        self.m[f'b{idx}'] = beta1 * self.m[f'b{idx}'] + (1 - beta1) * db
        self.v[f'b{idx}'] = self.beta2 * self.v[f'b{idx}'] + (1 - self.beta2) * db ** 2 

        # take timestep into account
        mt_w  = self.m[f'W{idx}'] / (1 - beta1 ** steps)
        vt_w  = self.v[f'W{idx}'] / (1 - self.beta2 ** steps)

        mt_b  = self.m[f'b{idx}'] / (1 - beta1 ** steps)
        vt_b  = self.v[f'b{idx}'] / (1 - self.beta2 ** steps)

        w_update = - self.learning_rate * mt_w / (np.sqrt(vt_w) + self.epsilon)
        b_update = - self.learning_rate * mt_b / (np.sqrt(vt_b) + self.epsilon)
        
        if self.weight_decay:
            gamma = self.gamma_init * self.decay_rate ** int(epoch_num / 5) 
            w_update = - self.learning_rate * mt_w / ((np.sqrt(vt_w) + self.epsilon) + gamma * layers[idx].W) 
            b_update = - self.learning_rate * mt_b / ((np.sqrt(vt_b) + self.epsilon) + gamma * layers[idx].b)

        layers[idx].W += w_update
        layers[idx].b += b_update

# Layer object to handle weights, biases, activation (if any) of a layer

class Layer:
    def __init__(self, hidden_units: int, activation:str=None):
        self.hidden_units = hidden_units
        self.activation = activation
        self.W = None
        self.b = None

    def initialize_params(self, n_in, hidden_units):
        # set seed for reproducibility
        np.random.seed(42)
        self.W = np.random.randn(n_in, hidden_units) * np.sqrt(2/n_in)
        np.random.seed(42)
        self.b = np.zeros((1, hidden_units))

    def forward(self, X):
        self.input = np.array(X, copy=True)
        if self.W is None:
            self.initialize_params(self.input.shape[-1], self.hidden_units)

        self.Z = X @ self.W + self.b

        if self.activation is not None:
            self.A = self.activation_fn(self.Z)
            return self.A
        return self.Z

    def activation_fn(self, z, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return self.relu(z, derivative=True)
            return self.relu(z, derivative=False)
        if self.activation == 'sigmoid':
            if derivative:
                return self.sigmoid(z, derivative=True)
            return self.sigmoid(z, derivative=False)
        if self.activation == 'softmax':
            if derivative:
                return self.dsoftmax(z)
            return self.softmax(z)

    def relu(self, x, derivative=False):
        '''
            Derivative of ReLU is a bit more complicated since it is not differentiable at x = 0

            Forward path:
            relu(x) = max(0, x)
            In other word,
            relu(x) = 0, if x < 0
                    = x, if x >= 0

            Backward path:
            ∇relu(x) = 0, if x < 0
                     = 1, if x >=0
        '''
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        '''
            Forward path:
            σ(x) = 1 / 1+exp(-z)

            Backward path:
            ∇σ(x) = exp(-z) / (1+exp(-z))^2
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        '''
            softmax(x) = exp(x) / ∑exp(x)
        '''
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def dsoftmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

    def __repr__(self):
        return str(f'''Hidden Units={self.hidden_units}; Activation={self.activation}''')

class FeedForward:
    def __init__(self):
        self.layers = dict()
        self.cache  = dict()
        self.grads  = dict()
        self.loss = []                      # cost list attribute
        
    def add(self, layer):
        self.layers[len(self.layers)+1] = layer

    def set_config(self, epochs, learning_rate, optimizer=None):
        self.epochs        = epochs
        self.optimizer     = optimizer
        self.learning_rate = learning_rate
        
        if not not self.optimizer:
            self.optimizer.config(self.layers)
            self.optimizer.epochs        = self.epochs
            self.optimizer.learning_rate = self.learning_rate
        
    # Function for forward propagation
    def forward(self, x):
        '''
            y = σ(wX + b)
        '''
        for idx, layer in self.layers.items():
            x = layer.forward(x)
            self.cache[f'W{idx}'] = layer.W
            self.cache[f'Z{idx}'] = layer.Z
            self.cache[f'A{idx}'] = layer.A
        return x

    # Back Propagation
    def backward(self, y):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        last_layer_idx = max(self.layers.keys())
        # number of examples
        m = y.shape[0]
        
        # back prop through all dZs
        for idx in reversed(range(1, last_layer_idx+1)):
            if idx == last_layer_idx:
                # e.g. dZ3 = y_pred - y_true for a 3 layer network
                self.grads[f'dZ{idx}'] = self.cache[f'A{idx}'] - y
            else:
                # dZn = dZ(n+1) dot W(n+1) * inverse derivative of activation function of Layer n, with Zn as input
                self.grads[f'dZ{idx}'] = self.grads[f'dZ{idx+1}'] @ self.cache[f'W{idx+1}'].T *\
                                        self.layers[idx].activation_fn(self.cache[f'Z{idx}'], derivative=True)
            self.grads[f'dW{idx}'] = 1 / m * self.layers[idx].input.T @ self.grads[f'dZ{idx}']
            self.grads[f'db{idx}'] = 1 / m * np.sum(self.grads[f'dZ{idx}'], axis=0, keepdims=True)

            assert self.grads[f'dW{idx}'].shape == self.cache[f'W{idx}'].shape
    
    def compute_loss(self, y_true, y_pred):
        '''
            L(y, ŷ) = −∑ylog(ŷ).
        '''
        l_sum = np.sum(y_true * np.log(y_pred))
        m = y_true.shape[0]
        loss = -(1./m)* l_sum
        return loss

    @staticmethod
    def mse(y_true, y_pred):
        m = y_true.shape[0]
        return np.mean((y_true - y_pred)**2)

    def optimize(self, key):
        # Vanilla minibatch gradient descent
        self.layers[key].W -= self.learning_rate * self.grads[f'dW{key}']
        self.layers[key].b -= self.learning_rate * self.grads[f'db{key}']

    def update_params(self, epoch_num, steps):
        for key in self.layers.keys():
            if self.optimizer is None:
                self.optimize(key)
            else:
                self.optimizer.optimize(key, self.layers, self.grads, epoch_num, steps)
    
    def __repr__(self):
        return str(self.layers)

    def check_accuracy(self, y_true, y_pred):
        c = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
        train_acc = list(c).count(True) / len(c) * 100
        return train_acc
        
    def fit(self, X_train, y_train, X_test=None, y_test=None,
              batch_size=32):
        '''Training cycle of the model object'''
        # Hyperparameters
        train_accs = []
        val_accs = []
        self.batch_size = batch_size
        num_batches = -(-X_train.shape[0] // self.batch_size)

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"

        # Train
        for epoch in range(self.epochs): # loop based on number of iterations
            # print(f'Epoch {epoch+1}')
            epoch_loss = []
            steps = 0
            
            # Shuffle
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuff = X_train[permutation]
            y_train_shuff = y_train[permutation]

            for batch_idx in range(num_batches):
                # Batch
                start_idx = batch_idx * self.batch_size
                end_idx   = min(start_idx + self.batch_size, X_train.shape[0]-1)
                x_batch = X_train_shuff[start_idx: end_idx]
                y_batch = y_train_shuff[start_idx: end_idx]

                # Forward
                steps += 1
                y_hat = self.forward(x_batch)
                loss  = self.compute_loss(y_batch, y_hat)
                epoch_loss.append(loss)
                
                # Backprop - calculation of gradients
                self.backward(y_batch)
                
                # Optimize / update weights and biases of each layer
                self.update_params(epoch+1, steps)
            
            # Cumpute Metrics(Accuracy) and Loss
            y_tra_hat  = self.forward(X_train)
            train_acc  = self.check_accuracy(y_train, y_tra_hat)
            
            train_loss = sum(epoch_loss) / len(epoch_loss)
            train_accs.append(train_acc)
            
            # store cost in list
            self.loss.append(train_loss)
            
            # Evaluate performance
            # Test data
            y_val_hat  = self.forward(X_test)
            val_acc    = self.check_accuracy(y_test, y_val_hat)
            val_loss   = self.compute_loss(y_test, y_val_hat)
            val_accs.append(val_acc)

            print(template.format(epoch+1, time.time()-start_time, train_acc, train_loss, val_acc, val_loss))

        self.history = {'train_loss': self.loss, 'train_acc': train_accs, 'val_acc': val_accs}
        
        print('Training Complete')
        print('----------------------------------------------------------------------------')

from sklearn import datasets, metrics, model_selection, preprocessing
# Load data
iris = datasets.load_iris()

X = np.array(iris.data)
y = np.array(iris.target)

# Reduce the sample size
from sklearn.model_selection import train_test_split
# train_size = 10000
# test_size  = 5000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshaping inputs for the our model
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

epochs      = 200
lr          = 1e-3
batch_size  = 40
lr_decay    = False
hidden_size_1 = 64
output_size = n_classes = len(np.unique(y_train))

# Function to convert labels to one-hot encodings
def one_hot(Y):
    # n_classes = len(set(Y))
    new_Y = []
    for label in Y:
        encoding = np.zeros(n_classes)
        encoding[label] = 1.
        new_Y.append(encoding)
    return np.array(new_Y)

"""
def one_hot(x, k, dtype=np.float32):
    # Create a one-hot encoding of x of size k.
    return np.array(x[:, None] == np.arange(k), dtype)
"""
y_train = one_hot(y_train)
y_test  = one_hot(y_test)

# Print data shape
print("Train/Test/Validation Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)


# ReLU + SGD
model = FeedForward()
model.add(Layer(hidden_size_1, activation='relu'))
model.add(Layer(hidden_size_1, activation='relu'))
model.add(Layer(n_classes, activation='softmax'))
model.set_config(epochs=epochs, learning_rate=lr, optimizer=None)

model.fit(X_train, y_train, X_test, y_test, batch_size=batch_size)

# plot the cost function
plt.grid()
plt.plot(range(model.epochs),model.loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Gradient Descent Loss Function')
plt.show()

optimizer = SGDM(lr, name='SGDM')

model = FeedForward()
model.add(Layer(hidden_size_1, activation='relu'))
model.add(Layer(hidden_size_1, activation='relu'))
model.add(Layer(n_classes, activation='softmax'))
model.set_config(epochs=epochs, learning_rate=lr, optimizer=optimizer)

model.fit(X_train, y_train, X_test, y_test, batch_size=batch_size)

# plot the cost function
plt.grid()
plt.plot(range(model.epochs),model.loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Momentum Loss Function')
plt.show()


optimizer = Adam(lr, weight_decay=True, name='AdamW')

model = FeedForward()
model.add(Layer(hidden_size_1, activation='relu'))
model.add(Layer(hidden_size_1, activation='relu'))
model.add(Layer(n_classes, activation='softmax'))
model.set_config(epochs=epochs, learning_rate=lr, optimizer=optimizer)

model.fit(X_train, y_train, X_test, y_test, batch_size=batch_size)

# plot the cost function
plt.grid()
plt.plot(range(model.epochs),model.loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Adam Loss Function')
plt.show()

