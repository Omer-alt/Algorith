import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

class SGD:
    
    def __init__(self) -> None:
        pass
    
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
        
    def train_with_sgd(self, X, y, num_epochs, step_size, plot_every=1):
        N, D = X.shape
        theta = self.initialize_theta(D)
        losses = []
        epoch = 0
        loss_tolerance = 0.001
        avg_loss = float("inf")
        
        while epoch < num_epochs and avg_loss > loss_tolerance:
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
                theta = self.update_function(theta, grads, step_size)
            
            # You can plot your data fitting here
            avg_loss = running_loss / X.shape[0]
            losses.append(avg_loss)
            print(f"Epoch {epoch}, loss {avg_loss}")
            
            epoch += 1
            
        return losses


# Prepared data for train model
xtrain = np.linspace(0,1, 10)
ytrain = xtrain + np.random.normal(0, 0.1, (10,))

xtrain = xtrain.reshape(-1, 1)
ytrain = ytrain.reshape(-1, 1)
plt.scatter(xtrain, ytrain, marker="+")
plt.show()
# Let's see the power of object oriented
sgd = SGD()

sgd_losses = sgd.train_with_sgd(xtrain, ytrain, 30, 0.1, 2)
sgd.plot_loss(sgd_losses)
plt.show()
            
            
            
            
            
            
            