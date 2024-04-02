#import required libaries
import numpy as np
import time
from matplotlib import pyplot as plt

class LogisticRegression():

    """
    This is the LogisticRegression class used to feedforward and backpropagate the network across a defined number
    of iterations and produce predictions. After iteration the predictions are assessed using
    Binary Cross Entropy Cost function.
    """

    print('Running...')
    
    def __init__(self, lr=1e-1, input_size = 30, hidden_size_1 = 9, output_size =1):
        # Hyperparameters
        self.lr = lr                        # learning rate attibute
        self.input_size    = input_size     # input layer attibute
        self.hidden_size_1 = hidden_size_1  # hidden layer attibute
        self.output_size   = output_size    # output layer attibute
        
        self.loss = []                      # cost list attribute
        self.y_hats = []                    # predictions list attribute
        
        # Save all weights
        self.initialize()

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
    
    def initialize(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size_1)  # weight attribute connecting to the hidden layer
        self.b1 = np.zeros((1, self.hidden_size_1))
        self.W2 = np.random.randn(self.hidden_size_1, self.output_size) # weight attribute connecting to the output layer
        self.b2 = np.zeros((1,  self.output_size))       
        
        
    # Function for forward propagation
    def forward(self, x):
        Z1 = np.dot(x, self.W1) + self.b1  # linear transformation to the hidden layer
        Z1 = np.clip(Z1, -10, 10)
        # print(Z1)
        # print(Z1.shape)
        A1 = self.sigmoid(Z1, derivative=False)    # hidden layer activation function
        # A1 = self.relu(Z1, derivative=False)       # hidden layer activation function
        
        Z2 = np.dot(A1, self.W2) + self.b2    # linear transformation to the output layer
        # print(Z2)
        Z2 = np.clip(Z2, -10, 10)
        y_hat = self.sigmoid(Z2, derivative=False)  # output layer prediction
        
        return Z1, A1, Z2, y_hat

    # Back Propagation
    def backward(self, Z1, A1, Z2, x, y, y_hat):
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
        m = y_hat.shape[0]
        
        # initiation of gradient descent algorithm
        
        dZ2 = y_hat - y

        dW2 = np.dot(A1.T, dZ2)    #∂loss/∂p *∂p/∂Z2 * ∂Z2/∂wh
        db2 = dZ2
        dA1 = np.multiply(dZ2, self.W2.T)
        
        dZ1 = np.multiply(dA1, self.sigmoid(Z1, derivative=True))
        # dZ1 = np.multiply(dA1, self.relu(Z1, derivative=True))
        dW1 = np.dot(x.T, dZ1)  #∂loss/∂p * ∂p/∂Z2 * ∂Z2/∂h * ∂h/∂z * ∂z/∂w
        db1 = dZ1
        
        return dW2, db2, dW1, db1
    
    def BCELoss(self, y_true, y_pred): # binary cross entropy cost function
        # binary cross entropy
        bce_loss = -(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))) / len(y_true)
        return bce_loss

    def update_params(self, dW2, db2, dW1, db1, l_rate=0.1):
        '''
            Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)

            Momentum:
            v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
            θ^(t+1) <- θ^t - ηv^(t+1)
        '''
        self.W1 = self.W1 - self.lr * dW1 
        self.b1 = self.b1 - self.lr * db1 
        self.W2 = self.W2 - self.lr * dW2 
        self.b2 = self.b2 - self.lr * db2
        
    def check_accuracy(self, y_true, y_pred):
        pred_labels = y_pred > 0.5
        accuracy = np.sum(y_true == pred_labels) / len(y_true)
        return accuracy
    
    def fit(self, X_train, y_train, X_test, y_test, epochs=100,
              batch_size=64, l_rate=0.1):  #train the network
        
        # Hyperparameters
        self.epochs = epochs
        
        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}"

        # Train
        for epoch in range(self.epochs): # loop based on number of iterations
            # print(f'Epoch {epoch+1}')
            
            # Forward
            Z1, A1, Z2, y_hat = self.forward(X_train)
            y_hat = np.clip(y_hat, 1e-9, 1-1e-9)
            
            # Cumpute Metrics(Accuracy) and Loss
            train_acc  = self.check_accuracy(y_train, y_hat)
            train_loss = self.BCELoss(y_train, y_hat)
            
            # store BCE cost in list
            self.loss.append(train_loss)
            
            # Back Propagation
            dW2, db2, dW1, db1 = self.backward(Z1, A1, Z2, X_train, y_train, y_hat)
            
            # Optimize / update weights and bias (gradient descent)
            self.update_params(dW2, db2, dW1, db1, l_rate=l_rate)
            
            print(template.format(epoch+1, time.time()-start_time, train_acc, train_loss))
            
        print('Training Complete')
        print('----------------------------------------------------------------------------')

from sklearn import datasets, metrics, model_selection, preprocessing

import numpy as np
import matplotlib.pyplot as plt

# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# Create dataset
N = 500
X, y = spiral_data(samples=N, classes=2)

# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data For Better Training and also for avoiding Zero Division Errors in the beggining of learning
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std  = std_scale.transform(X_test)

# Reshaping inputs for the our model
# X_train = X_train.T
# X_test = X_test.T
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)

epochs      = 100
batch_size  = 50
input_size  = X_train.shape[1]
hidden_size_1 = 64

# Print data shape
print("Train/Test/Validation Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

model = LogisticRegression( input_size = input_size, hidden_size_1 = hidden_size_1 )  # Pass data to the model (design matrix and y label)
model.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size, l_rate=0.05)  # Train the model

# plot the cost function
plt.grid()
plt.plot(range(model.epochs),model.loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('BCE Loss Function')
plt.show()

