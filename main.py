import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.datasets import make_classification

from linear_Regression import Linear_regression
from sgd_oop import SGD
from logistic_regression import Logistic_Regression
from sgd_with_momentum import SgdMomentum
from adam_gradient_descent import AdamGD

# class Singleton(Linear_regression, SGD):
#     _singleton = {}
    
#     def __new__(cls):
#         if not hasattr(cls, 'singleton'):
#             return super().__new__()
#         return cls._singleton

def train_test_split(X, y):
    
    np.random.seed(0)
    train_size = 0.8
    n = int(len(X)*train_size)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_idx = indices[: n]
    test_idx = indices[n: ]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, y_train, X_test, y_test                
                 
    
def main():
    
    
    """
    ____Prepared data for train model 
    - Gradient descent, 
    - stochastic gradient descent, 
    - stochastic gradient descent with momentum, 
    - Adam Gradient descent,
    - Minibatch gradient descent 
    """
    menu = ["- Gradient descent",  "- stochastic gradient descent,", "- stochastic gradient descent with momentum,", "- Adam Gradient descent,", "- Minibatch gradient descent" ]
    
    print("_________ Hello ___________", end="\n")
    [print(i, menu[i], end="\n") for i in range(len(menu))]
    
    choice = int(input("choose what you want to run: "))
    while choice > len(menu):
        choice = int(input(f"choose what you want to run from 0 to: {len(menu) - 1}"))
    
    # Data for linear regression trainning
    xtrain = np.linspace(0, 1, 10)
    ytrain = xtrain + np.random.normal(0, 0.1, (10,))

    xtrain = xtrain.reshape(-1, 1)
    ytrain = ytrain.reshape(-1, 1)
    
    # Data for logistique training ( Load the iris dataset from sklearn )
    X,y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    
    # instance of class selected
    instance = None
    
    match choice :
        case 0:
            instance = Linear_regression(lr=0.1, n_epochs=60)
            # Model Training
            print("0 - Gradient descent: ")
            instance.train_gd(xtrain, ytrain)
            
        case 1:
            print("1 - Stochastic Gradient descent : ")
            instance = SGD(lr=0.1, n_epochs=30)
            instance.train_with_sgd(xtrain, ytrain, 2)
            
        case 2:
            print("2 - Stochastic Gradient descent with momentum : ")
            instance = SgdMomentum(lr=0.1, n_epochs=30)
            instance.train_sgd_with_momentum (X=xtrain, y=ytrain, beta=0.99)
            
        case 3:
            print("3 - Adam Gradient descent : ")
            instance = AdamGD(0.1, 10)
            instance.adam_gradient_descent(xtrain, ytrain, 2) 
        case 4:
            print("4 - Mini batch Gradient descent : ")
            pass
        case 5:
            print("5 - Logistic regression : ")
            instance = Logistic_Regression ( 0.01, n_epochs=10000 )
            instance.fit(X_train,y_train)
            
    
    if instance:
        plt.plot(instance.train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    
if __name__ == "__main__":
    main()
