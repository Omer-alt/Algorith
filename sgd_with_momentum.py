import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

class SgdMomentum:
    def __init__(self, lr, n_epochs) :
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_losses = []
    
    def initialized_theta(self, D):
        return np.zeros((D, 1))
    
    def shuffled_data(self, X, y):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        return X[shuffled_idx], y[shuffled_idx]
    
    def linear_function(self, X, theta):
        assert X.ndim > 1
        assert X.shape[1] == theta.shape[0], f"The X number of cols X.COLS ={X.shape[1]} is different of the number of line of theta, theta.lines = {theta.shape[0]}" 
        
        return X @ theta
    
    def plot_loss(self, losses):
        """
            Plotting function for losses
        """
        plt.subplot(2,1,2)
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("training curve")
    
    def mean_squared_error(self, ytrue, ypred):
        return np.sum(np.square(ytrue - ypred)) / ytrue.shape[0]
    
    def per_sample_gradient(self, xi, yi, theta):
        return 2 * xi.T * (self.linear_function(xi, theta) - yi)
    
    def update_function(self, theta, momentum, step_size):
        return theta - momentum * step_size
        
    
    def get_momentum (self, momentum, grad, beta):
        return momentum * beta + (1 - beta) * grad
    
    
    def train_sgd_with_momentum(self, X, y, beta, plot_every=1 ):
        
        N, D = X.shape
        theta = self.initialized_theta(D)
        avg_losse = float("inf")
        loss_tolerance = 0.001
        epoch = 0.0
         
        while epoch < self.n_epochs and avg_losse > loss_tolerance :
            momentum = 0.0
            running_losse = 0.0
            shuffled_x, shuffled_y = self.shuffled_data(X, y)
            
            for idx in range(shuffled_x.shape[0]):
                sample_x = shuffled_x[idx].reshape(-1, D)
                sample_y = shuffled_y[idx].reshape(-1, 1)
                
                y_pred = self.linear_function(sample_x, theta)
                loss = self.mean_squared_error(sample_y, y_pred)
                running_losse += loss
                grads = self.per_sample_gradient(sample_x, sample_y, theta)
                momentum = self.get_momentum(momentum, grads, beta)
                theta = self.update_function(theta, momentum, self.lr)
                
            # Plot your scatter and prediction
            avg_losse = running_losse / X.shape[0]
            self.train_losses.append(avg_losse)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, loss {avg_losse}")
            
            epoch += 1
            
        return self.train_losses
        

                
                
        
        
    
    
    
    
    
    
    