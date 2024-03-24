import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
from sklearn.datasets import load_iris

class PCA:
  def __init__(self, n_component):
    self.n_component = n_component
    self.X_std = None
    self.eigen_vectors_sorted = None
    self.eigen_values_sorted = None
    self.explained_variance = None


  def mean(self, X):
    mean = np.sum(X, axis = 0) / X.shape[0]
    return mean

  def std(self, X):
    std = np.sqrt( np.sum((X - self.mean(X))** 2, axis = 0) /( X.shape[0] -1) )
    return std

  def Standardize_data(self, X):

    X_std = (X - self.mean(X)) / self.std(X)
    return X_std

  def covariance(self, X):

    cov = X.T @ X /( X.shape[0] - 1)
    return cov

  def explain_ratio(self, cov_mat):
    # compute eigen value and eigen vectors, rank them
    eigen_values, eigen_vectors = eig( cov_mat )
    idx = np.array([ np.abs(i) for i in eigen_values ]).argsort()[ ::-1 ]
    self.eigen_values_sorted = eigen_values [ idx ]
    self.eigen_vectors_sorted = eigen_vectors.T [ : , idx ]
    explained_variance = [(i / sum(eigen_values))*100 for i in self.eigen_vectors_sorted]
    self.explained_variance = np.round(explained_variance, 2)

  def fit(self, X):
    self.X_std = self.Standardize_data(X)
    cov_mat = self.covariance(self.X_std)
    self.explain_ratio(cov_mat)


  def transform(self):

    P = self.eigen_vectors_sorted[: self.n_component, :]
    assert self.X_std.shape[1] ==  P.T.shape[0]
    X_proj = self.X_std.dot(P.T)

    return X_proj


# Data loader
iris = load_iris()
X = iris['data']
y = iris['target']

if __name__ == "__main__":
    my_pca = PCA(n_component=2)
    my_pca.fit(X)
    new_X = my_pca.transform()
    
    plt.title(f"PC1 vs PC2")
    plt.scatter(new_X[:, 0], new_X[:, 1], c=y)
    plt.xlabel('PC1'); plt.xticks([])
    plt.ylabel('PC2'); plt.yticks([])
    plt.show()


