#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

class FeedForward():

    """
    This is the FeedForward class used to feedforward and backpropagate the network across a defined number
    of iterations and produce predictions. After iteration the predictions are assessed using
    Categorical Cross Entropy Cost function.
    """

    print('Running...')
    
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes

        # Choose activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")
        
        self.loss = []                      # cost list attribute
        self.y_hats = []                    # predictions list attribute
        
        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}

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
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        # number of nodes in each layer
        input_size    = self.sizes[0]
        hidden_size_1 = self.sizes[1]
        output_size   = self.sizes[2]

        params = {
            "W1": np.random.randn(hidden_size_1, input_size) * np.sqrt(1./input_size),
            "b1": np.zeros((hidden_size_1, 1)) * np.sqrt(1./input_size),
            "W2": np.random.randn(output_size, hidden_size_1) * np.sqrt(1./hidden_size_1),
            "b2": np.zeros((output_size, 1)) * np.sqrt(1./hidden_size_1)
        }
        return params
        
    def initialize_momemtum_optimizer(self):
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
        }
        return momemtum_opt
        
    # Function for forward propagation
    def forward(self, x):
        '''
            y = σ(wX + b)
        '''
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        return self.cache["A2"]

    # Back Propagation
    def backward(self, y, y_hat):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        
        # number of examples
        m = y.shape[0]
        
        # initiation of gradient descent algorithm        
        dZ2 = y_hat - y.T
        dW2 = (1./m) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.matmul(self.params["W2"].T, dZ2)
        
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1./m) * np.matmul(dZ1, self.cache["X"])
        db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)
        
        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads
    
    def compute_loss(self, y_true, y_pred):
        '''
            L(y, ŷ) = −∑ylog(ŷ).
        '''
        l_sum = np.sum(np.multiply(y_true.T, np.log(y_pred)))
        m = y_true.shape[0]
        loss = -(1./m)* l_sum
        return loss

    def update_params(self, l_rate=0.1, beta=.9):
        '''
            Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)

            Momentum:
            v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
            θ^(t+1) <- θ^t - ηv^(t+1)
        '''
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + (1. - beta) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Optimizer is currently not support, please use 'sgd' or 'momentum' instead.")
        
    def check_accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true, axis=-1) == np.argmax(y_pred.T, axis=-1))
    
    def fit(self, X_train, y_train, X_test, y_test, epochs=100,
              batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9):  #train the network
        
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-X_train.shape[0] // self.batch_size)
        
        # Initialize optimizer
        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.momemtum_opt = self.initialize_momemtum_optimizer()

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"

        # Train
        for epoch in range(self.epochs): # loop based on number of iterations
            # print(f'Epoch {epoch+1}')
            
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
                y_hat = self.forward(x_batch)
                
                # Backprop - calculation of gradients
                grad = self.backward(y_batch, y_hat)
                
                # Optimize / update weights and biases of each layer
                self.update_params(l_rate=l_rate, beta=beta)

            # Cumpute Metrics(Accuracy) and Loss
            y_hat  = self.forward(X_train)
            train_acc  = self.check_accuracy(y_train, y_hat)
            train_loss = self.compute_loss(y_train, y_hat)
            
            # store cost in list
            self.loss.append(train_loss)
            
            # Test data
            y_val_hat  = self.forward(X_test)
            val_acc    = self.check_accuracy(y_test, y_val_hat)
            val_loss   = self.compute_loss(y_test, y_val_hat)
            
            print(template.format(epoch+1, time.time()-start_time, train_acc, train_loss, val_acc, val_loss))
                
        print('Training Complete')
        print('----------------------------------------------------------------------------')

from sklearn import datasets, metrics, model_selection, preprocessing
# Load data
iris = datasets.load_iris()

X = np.array(iris.data)
y = np.array(iris.target)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Reshaping inputs for the our model
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

epochs      = 200
batch_size  = 40
input_size  = X_train.shape[1]
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

# Sigmoid + Momentum
model = FeedForward(sizes=[input_size, hidden_size_1, output_size], activation='sigmoid')
model.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, optimizer='momentum', l_rate=0.05, beta=.9)

# plot the cost function
plt.grid()
plt.plot(range(model.epochs),model.loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Sigmoid + Momentum Loss Function')
plt.show()

# ReLU + SGD
model = FeedForward(sizes=[input_size, hidden_size_1, output_size], activation='relu')
model.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, optimizer='sgd', l_rate=0.05)

# plot the cost function
plt.grid()
plt.plot(range(model.epochs),model.loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ReLU + SGD Loss Function')
plt.show()

