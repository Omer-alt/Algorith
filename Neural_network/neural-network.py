import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self) -> None:
        pass
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp (-Z))
    
    def d_sigmoid( self, Z ):
        return self.sigmoid(Z)*(1 - self.sigmoid(Z))
    
class NeuronLayer (Neuron):
    def __init__(self, n_inputs, n_neurons) :
        super().__init__() 
        # to configure in ititialisation
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        
        # to update during computation
        self.Z = None
        self.A = None
        self.dL_dA = None # is self.dZ_dA but for the last layer
        self.dZ_dA = None 
        self.dA_dZ = None
        self.dZ_dW = None
        self.dW = None
        self.db = None
        
        # make initialization by default
        self.W = np.random.randn(n_neurons, n_inputs)
        self.b = np.random.randn(n_neurons, 1)
        
    def forward_pass(self, X):
        self.Z = self.W.dot(X) + self.b
        self.A = self.sigmoid(self.Z)
        
    def backward_pass(self):
        
        # Your code here
        
        self.dA_dZ = self.d_sigmoid(self.Z)
        self.dZ_dW = self.A.T
        
class NeuralNetWork:
    # number of node in layer
    h0, h1, h2 = 2, 10, 1
    
    def __init__(self, alpha=0.0001, n_epochs=10000):
        # Initialization With knowledge of my neural network
        # Neural Layers
        self.NL0 = NeuronLayer(1, 2)
        self.NL1 = NeuronLayer(self.h0, self.h1)
        self.NL2 = NeuronLayer(self.h1, self.h2)
        
        self.alpha = alpha
        self.n_epochs = n_epochs
        
    def forward_pass(self, X): 
        self.NL1.forward_pass(X) #Compute Z1, A1
        self.NL2.forward_pass(self.NL1.Z) #Compute Z2 and A2
        
    
    def backward_pass(self, X, Y):
        # last layer
        self.NL2.dL_dA = (self.NL2.A - Y) / (self.NL2.A * (1 - self.NL2.A))
        self.NL2.dA_dZ = self.NL2.sigmoid(self.NL2.Z)
        self.NL2.dZ_dW = self.NL1.A.T
        
        self.NL2.dW = (self.NL2.dL_dA * self.NL2.dA_dZ) @ self.NL2.dZ_dW
        self.NL2.db = self.NL2.dL_dA @ self.NL2.dA_dZ.T
        
        # second layer
        self.NL1.dZ_dA = self.NL2.W # dZ2_dA1 = W2
        self.NL1.dA_dZ = self.NL1.sigmoid(self.NL1.Z)
        self.NL1.dZ_dW = X.T # can be optimized if we give to self.NL0.A = X
        
        self.NL1.dW = ( self.NL1.dZ_dA.T * (self.NL2.dL_dA * self.NL2.dA_dZ) * self.NL1.dA_dZ  ) @ self.NL1.dZ_dW 
        self.NL1.db = ((self.NL2.dL_dA * self.NL2.dA_dZ) @ (self.NL1.dZ_dA.T * self.NL1.dA_dZ).T).T

    def accuracy(self, y_pred, y):

        # Your code here
        # y = y.reshape(-1, 1)
        # y_pred = y_pred.reshape(-1, 1)
        y_pred = (y_pred > 0.5).astype(int)
        correct_predictions = (y_pred == y)
        return np.mean(correct_predictions)

    
    def predict(self, X):

        # Your code here
        self.forward_pass(X)
        return self.NL2.A
    
    def update(self):
        self.NL1.W -= self.alpha * self.NL1.dW
        self.NL2.W -= self.alpha * self.NL2.dW
        
        self.NL1.b -= self.alpha * self.NL1.db
        self.NL2.b -= self.alpha * self.NL2.db

    def loss(self, y_pred, Y):
        return  np.divide(-np.sum(Y * np.log(y_pred)+(1-Y)* np.log(1-y_pred)),Y.shape[1])
    
    def train (self, X_train, X_test, Y_train, Y_test):
        train_loss = []
        test_loss = []
        
        for i in range(self.n_epochs):
            
            self.forward_pass(X_train)
            self.backward_pass(X_train, Y_train)
            self.update()
            
            # save the train loss
            train_loss.append(self.loss(self.NL2.A, Y_train))
            
            # save test loss
            self.forward_pass(X_test)
            test_loss.append(self.loss(self.NL2.A, Y_test))
            
            ## plot boundary
            if i % 50 == 0:
                self.plot_decision_boundary()
                
        return train_loss, test_loss
                
    def plot_decision_boundary(self):
        x = np.linspace(-0.5, 2.5,100 )
        y = np.linspace(-0.5, 2.5,100 )
        xv , yv = np.meshgrid(x,y)
        xv.shape , yv.shape
        X_ = np.stack([xv,yv],axis = 0)
        X_ = X_.reshape(2,-1)
        self.forward_pass(X_)
        plt.figure()
        plt.scatter(X_[0,:], X_[1,:], c= self.NL2.A)
        plt.show()
            
            
#   Action on data

# generate data
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4,2)
class_0_b =var * np.random.randn(n//4,2) + (2,2)

class_1_a = var* np.random.randn(n//4,2) + (0,2)
class_1_b = var * np.random.randn(n//4,2) +  (2,0)

X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
X.shape, Y.shape

# shuffle the data
rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]

X = X.T
Y = Y.T

# train test split
ratio = 0.8
X_train = X [:, :int (n*ratio)]
Y_train = Y [:, :int (n*ratio)]

X_test = X [:, int (n*ratio):]
Y_test = Y [:, int (n*ratio):]





if __name__ == "__main__":
    # initialise the Network
    
    # Visualize first your data
    plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
    plt.show()
    
    
    nn = NeuralNetWork(0.1, 200)
    train_loss, test_loss = nn.train(X_train, X_test, Y_train, Y_test)
    
    ## plot train et test losses
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.show()
    
    y_pred = nn.predict(X_train)
    train_accuracy = nn.accuracy(y_pred, Y_train)
    print ("train accuracy :", train_accuracy)

    y_pred = nn.predict(X_test)
    test_accuracy = nn.accuracy(y_pred, Y_test)
    print ("test accuracy :", test_accuracy)
                