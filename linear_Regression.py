import numpy as np
import matplotlib.pyplot as plt

class Linear_regression:
    
  def __init__(self, lr, n_epochs ):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
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

  def train_gd(self, X, y):
    
    N, D = X.shape
    self.w = self.initilize_theta(D)

    for epoch in range (self.n_epochs):
      ypred = self.linear_function(X)
      loss = self.mse(y, ypred)
      grad = self.grad_mse(X, y)
      self.w = self.update_function(grad)

      self.train_losses.append(loss)
      if epoch % 5 == 0:
        print (f"Epoch {epoch} loss {loss}")

    return self.train_losses
