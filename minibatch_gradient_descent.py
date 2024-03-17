import numpy as np

class MiniBatchGD:
    
  def __init__(self, lr, n_epochs ):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.batch_size = 3
    self.weight = []

  def initilize_theta(self, D):
    return np.zeros((D, 1))

  def linear_function(self, X):
    
    assert X.ndim > 1
    assert X.shape[1] == self.w.shape[0]

    return X @ self.w

  def mse(self, y, ypred):

    return np.sum(np.square(y - ypred)) / y.shape[0]

  def grad_mse (self, X, y):
    return 2 * X.T @ (self.linear_function(X) - y) / y.shape[0]

  def update_function(self, grad):
    return self.w - self.lr * grad

  def shuffled_data(self, X, y):
    N, _ = X.shape
    shuffled_idx = np.random.permutation(N)
    return X[shuffled_idx], y[shuffled_idx]

  def train_minibatch_gd(self, X, y):
    
    N, D = X.shape
    self.w = self.initilize_theta(D)
    num_batches = N//self.batch_size
    shuffle_x, shuffle_y = self.shuffled_data(X, y)
    
    for epoch in range(self.n_epochs):
        running_loss = 0.0
    
        for batch_idx in range(0, N, self.batch_size):
            x_batch = shuffle_x[batch_idx: batch_idx + self.batch_size] 
            y_batch = shuffle_y[batch_idx: batch_idx + self.batch_size]
            
            ypred = self.linear_function(x_batch)
            loss = self.mse(y_batch, ypred)
            running_loss += (loss * x_batch.shape[0])
            grad = self.grad_mse(x_batch, y_batch)
            self.w = self.update_function(grad)

        avg_loss = running_loss / N
        self.train_losses.append(avg_loss)
        if epoch % 5 == 0:
            print (f"Epoch {epoch} loss {loss}")

    return self.train_losses
