import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Logistic_Regression:
    def __init__(self, lr, n_epochs) :
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_losses = []
        self.w = None
        self.weight = []
    
    def sigmoid(self, x ):
        
        assert x.shape[1] == self.w.shape[0], f"Dimension mis match"
        z = x @ self.w
        return 1 / (1 + np.exp(-z)) 
    
    def add_ones(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x))
    
    def cross_entropy(self, x, y_true):
        y_pred = self.sigmoid(x)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) ) / y_true.shape[0]
    
    def predict_proba(self, x):
        proba = self.sigmoid(x)
        return proba
    
    def predict(self, x):
        probas = self.predict_proba(x)
        output = [1 if y >= 0.5 else 0 for y in probas ]
        return output
    
    def fit(self, x, y):
        # Default settings
        x = self.add_ones(x)
        N, D = x.shape
        y = y.reshape(-1, 1)
        self.w = np.zeros((D, 1))
        
        for epoch in range(self.n_epochs):
            # predictions
            ypred = self.predict_proba(x)
           
            # compute the gradient
            grad = -x.T @ (y - ypred) / N
            # update rule
            self.w -= self.lr * grad 
            # compute loss
            loss = self.cross_entropy(x, y)
            self.train_losses.append(loss)
            
            if epoch % 1000 == 0:
                print(f"loss for epoch {epoch} : {loss}")
                
                
# split data function 
# def train_test_split(X, y):
    
#     np.random.seed(0)
#     train_size = 0.8
#     n = int(len(X)*train_size)
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     train_idx = indices[: n]
#     test_idx = indices[n: ]
#     X_train, y_train = X[train_idx], y[train_idx]
#     X_test, y_test = X[test_idx], y[test_idx]
    
#     return X_train, y_train, X_test, y_test                
                
# # Load the iris dataset from sklearn 
# X,y = make_classification(n_features=2, n_redundant=0,
#                           n_informative=2, random_state=1, 
#                           n_clusters_per_class=1)


# X_train, y_train, X_test, y_test = train_test_split(X, y)
 
# model = Logistic_Regression ( 0.01, n_epochs=10000 )
# model.fit(X_train,y_train)

# plt.plot(model.train_losses)
# plt.show()

