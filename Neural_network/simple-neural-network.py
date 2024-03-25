import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, bias):
        self.bias = bias
    
    
class NeuronLayer :
    def __init__(self, n_inputs, n_neurons) :
        # to configure in ititialisation
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.neurons = []


    def inspect(self):

        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            print('  Bias:', self.neurons[n].bias)

    def sigmoid(self, z):
        return 1 / (1 + np.exp (-z)) 

    def d_sigmoid(self, z):

        s = self.sigmoid(z)
        return s * (1 - s)

    def init_params(self):

        W, b = np.random.randn(self.n_neurons, self.n_inputs), np.random.randn(self.n_neurons, 1)  

        for i in range(self.n_neurons): 
            new_neuron = Neuron(b[i])
            self.neurons.append(new_neuron)

        return W, b
        
    def forward_pass(self, X, W, b):
        Z = W.dot(X) + b
        A = self.sigmoid(Z)
        return A, Z
        

class NeuralNetwork:
    def __init__(self, alpha, n_epochs) :
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.h0 = 2
        self.h1 = 10
        self.h2 = 1

        self.hidden_layer = NeuronLayer(self.h0, self.h1)
        self.output_layer = NeuronLayer(self.h1, self.h2)
    
    def loss(self, y_pred, Y):
        return  np.divide(-np.sum(Y * np.log(y_pred)+(1-Y)* np.log(1-y_pred)),Y.shape[1])
    
    def forward_pass(self, X, W1, W2, b1, b2):
        
        Z1 = W1.dot(X) + b1
        A1 = self.hidden_layer.sigmoid(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.output_layer.sigmoid(Z2)
        return A2, Z2, A1, Z1
    
    def backward_pass(self, X, Y, A2, Z2, A1, Z1, W1, W2, b1, b2):

        dL_dA2 = (A2-Y)/(A2*(1-A2))
        dA2_dZ2 = self.output_layer.d_sigmoid(Z2)
        dZ2_dW2 = A1.T

        dW2 = ( dL_dA2 * dA2_dZ2  ) @ dZ2_dW2
        db2 = dL_dA2 @ dA2_dZ2.T

        dZ2_dA1 = W2
        dA1_dZ1 = self.hidden_layer.d_sigmoid(Z1)
        dZ1_dW1 = X.T

        dW1 = ( dZ2_dA1.T * (dL_dA2 * dA2_dZ2) * dA1_dZ1 ) @ dZ1_dW1
        db1 = ((dL_dA2 * dA2_dZ2) @ (dZ2_dA1.T * dA1_dZ1).T).T

        return dW1, dW2, db1, db2
    
    def accuracy(self, y_pred, y):

        # y = y.reshape(-1, 1)
        # y_pred = y_pred.reshape(-1, 1)
        y_pred = (y_pred > 0.5).astype(int)
        correct_predictions = (y_pred == y)
        return np.mean(correct_predictions)


    def predict(self, X,W1,W2, b1, b2):

        A2, _, _, _ = self.forward_pass(X, W1, W2, b1, b2)
        return A2
    
    def update(self, W1, W2, b1, b2,dW1, dW2, db1, db2 ):

        # Your code here
        W1 -= self.alpha * dW1
        W2 -= self.alpha * dW2
        b1 -= self.alpha * db1
        b2 -= self.alpha * db2

        return W1, W2, b1, b2
    
    def plot_decision_boundary(self, W1, W2, b1, b2):
        x = np.linspace(-0.5, 2.5,100 )
        y = np.linspace(-0.5, 2.5,100 )
        xv , yv = np.meshgrid(x,y)
        xv.shape , yv.shape
        X_ = np.stack([xv,yv],axis = 0)
        X_ = X_.reshape(2,-1)
        A2, Z2, A1, Z1 = self.forward_pass(X_, W1, W2, b1, b2)
        plt.figure()
        plt.scatter(X_[0,:], X_[1,:], c= A2)
        plt.show()
        
    def train(self, X_train, Y_train, X_test, Y_test ):
        # W1, W2, b1, b2 = self.init_params()
        W1, b1 = self.hidden_layer.init_params()
        W2, b2 = self.output_layer.init_params()

        train_loss = []
        test_loss = []
        
        for i in range(self.n_epochs):
            ## forward pass
            A2, Z2, A1, Z1 = self.forward_pass(X_train, W1, W2, b1, b2)
            ## backward pass
            dW1, dW2, db1, db2 = self.backward_pass(X_train, Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
            ## update parameters
            W1, W2, b1, b2 = self.update(W1, W2, b1, b2, dW1, dW2, db1, db2)

            ## save the train loss
            train_loss.append(self.loss(A2, Y_train))
            ## compute test loss
            A2, Z2, A1, Z1 = self.forward_pass(X_test, W1, W2, b1, b2)
            test_loss.append(self.loss(A2, Y_test))

            ## plot boundary
            if i %1000 == 0:
                self.plot_decision_boundary(W1, W2, b1, b2)
        
        return train_loss, test_loss, W1, W2, b1, b2

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(3))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')
        
 
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
    # Visualize first your data
    plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
    plt.show()   
    
    nn = NeuralNetwork(alpha=0.001, n_epochs=10000)  
    
    train_loss, test_loss, W1, W2, b1, b2 = nn.train(X_train, Y_train, X_test, Y_test)
    
    ## plot train et test losses
    plt.plot(train_loss, label="L(Y,Y_train)")
    plt.plot(test_loss, label="L(Y,Y_test)")
    plt.legend(loc='upper right')
    plt.title('lr = {} , n_epoch = {}'.format(nn.alpha, nn.n_epochs ), fontstyle='oblique')
    plt.show()
    
    y_pred = nn.predict(X_train, W1,W2, b1, b2 )
    train_accuracy = nn.accuracy(y_pred, Y_train)
    print ("train accuracy :", train_accuracy)

    y_pred = nn.predict(X_test, W1,W2, b1, b2)
    test_accuracy = nn.accuracy(y_pred, Y_test)
    print ("test accuracy :", test_accuracy) 

    # Consult the structure of Neural network build
    nn.inspect()