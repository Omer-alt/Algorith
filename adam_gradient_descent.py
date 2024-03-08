import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

class AdamGD:
    def __init__(self, lr, epoch) :
        self.lr = lr
        self.epoch = epoch
        self.theta = None
        self.train_losses = []
        self.weight = []
        
    def initialize_theta(self, D):
        return np.zeros((D, 1))
    
    def linear_function(self, X, theta):
        assert X.ndim > 1
        assert theta.ndim > 1
        
        assert X.shape[1] == theta.shape[0], f"The number of columns of X {X.shape[1]} is equal of {theta.shape[0]} " 
        
        return X @ theta
        
    def mean_squared_error(self, ytrue, ypred):
        """
        Computes the mean squared error
        Args:
            ytrue: vector of true labels
            ypred: vector of predicted labels

        Returns:
            mse loss (scalar)
        """
        return (np.sum(np.square(ytrue - ypred))) / ytrue.shape[0]
    
    def plot_loss(self, losses):
        """
            Plotting function for losses
        """
        plt.subplot(2,1,1)
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("training curve")
        plt.show()
    
    def batch_gradient(self, X, y, theta):
        """Computes gradients of loss wrt parameters for a full batch
        Args:
            X: input features of size - N x D
            y: target vector of size - N x 1
            theta: parameters of size - D x 1
        """
        return 2 * X.T @ (self.linear_function( X , theta) - y) / y.shape[0]
    
    # Update parameters using Adam
    def update_parameters_with_adam(self, theta, grads, m, v, epoch, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * grads ** 2
        
        m_hat = m / (1.0 - beta1 ** (epoch + 1))
        v_hat = v / (1.0 - beta2 ** (epoch + 1))
        
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
        return theta, m, v
        
    def adam_gradient_descent (self, X, y, num_epochs, plot_every=1):
        """
            Trains model with adam  gradient descent
            Args:
                X: feature matrix (size - N x D)
                y: target (size - N x 1)
                num_epochs: iterations ( int )
            Returns:
                output losses, size num_epochs x 1
        """
        N, D = X.shape
        theta = self.initialize_theta(D)
        losses = []
        m = np.zeros(D)
        v = np.zeros(D)

        for epoch in range (num_epochs):
            ypred = self.linear_function(X, theta)
            loss = self.mean_squared_error(y, ypred)
            losses.append(loss)
            grads = self.batch_gradient(X, y, theta)
            theta, m, v = self.update_parameters_with_adam(theta, grads, m, v, epoch, self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
            
            if epoch % 5 == 0 :
                print(f"\nEpoch {epoch}, loss {loss}")

        return losses
    
    
# Prepared data for train model
xtrain = np.linspace(0,1, 10)
ytrain = xtrain + np.random.normal(0, 0.1, (10,))

xtrain = xtrain.reshape(-1, 1)
ytrain = ytrain.reshape(-1, 1)
plt.subplot(2,1,2)
plt.scatter(xtrain, ytrain, marker="+")
plt.show()
# Let's see the power of object oriented
adam_gd = AdamGD(0.1, 10)

sgd_losses = adam_gd.adam_gradient_descent(xtrain, ytrain, 30, 2)
adam_gd.plot_loss(sgd_losses)