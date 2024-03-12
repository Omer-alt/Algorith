import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

class SGD:
    
    def __init__(self, lr, n_epochs) :
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_losses = []
    
    def initialize_theta(self, D):
        return np.zeros((D, 1))
    
    def linear_function (self, xi, theta):
        assert xi.shape[1] == theta.shape[0],f"the number of columns of X:{xi.shape[1]} is different from the number of rows of theta {theta.shape[0]} "
        return xi @ theta
    
    def per_sample_gradient(self, xi, yi, theta):
        return 2 * xi.T * (self.linear_function(xi, theta) - yi)
    
    def update_function(self, theta, grads, step_size):
        return theta - grads*step_size
    
    def shuffle_data(self, X, y):
        N, _=X.shape
        shuffle_idx = np.random.permutation(N)
        return X[shuffle_idx], y[shuffle_idx]
    
    def mean_squered_error(self, y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2) / y_true.shape[0]
    
    def plot_loss(self, losses):
        """
            Plotting function for losses
        """
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("training curve")
        
    def train_with_sgd(self, X, y, plot_every=1):
        N, D = X.shape
        theta = self.initialize_theta(D)
        epoch = 0
        loss_tolerance = 0.001
        avg_loss = float("inf")
        
        while epoch < self.n_epochs and avg_loss > loss_tolerance:
            running_loss = 0.0
            shuffled_x, shuffled_y = self.shuffle_data(X, y)
            # if we take idx in X.shape it will work ?? yess
            for idx in range(shuffled_x.shape[0]):
                sample_x = shuffled_x[idx].reshape(-1, D)
                sample_y = shuffled_y[idx].reshape(-1, 1)
                ypred = self.linear_function(sample_x, theta)
                loss = self.mean_squered_error(sample_x, ypred)
                running_loss += loss
                grads = self.per_sample_gradient(sample_x, sample_y, theta)
                theta = self.update_function(theta, grads, self.lr)
            
            # You can plot your data fitting here
            avg_loss = running_loss / X.shape[0]
            self.train_losses.append(avg_loss)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, loss {avg_loss}")
            
            epoch += 1
            
        return self.train_losses


            
            
            
            
            
            
            