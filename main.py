import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.datasets import make_classification
from ipywidgets import interact, FloatSlider, IntSlider, fixed

from linear_Regression import Linear_regression
from minibatch_gradient_descent import MiniBatchGD
from sgd_oop import SGD
from logistic_regression import Logistic_Regression
from sgd_with_momentum import SgdMomentum
from adam_gradient_descent import AdamGD

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

#Try to have graph on playing with hyperparameters
# def plot_loss(lr, n_epochs, reg_class, xtrain, ytrain):
#     """"
#     Plot dynamique graph
#     Args:
#         lr, n_epochs: dynamically changeable parameters
#         reg_class: The class to instanciate for the regression
#         xtrain, ytrain: data for training our model
#     Returns:
#         graph  
#     """
#     instance = reg_class(lr=lr, n_epochs=n_epochs)
#     instance.train_gd(xtrain, ytrain)
#     plt.plot(instance.train_losses)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss')
#     plt.show()  

class PlotHandler:
    
    # Data for linear regression trainning
    xtrain = np.linspace(0, 1, 10)
    ytrain = xtrain + np.random.normal(0, 0.1, (10,))

    xtrain = xtrain.reshape(-1, 1)
    ytrain = ytrain.reshape(-1, 1)
    
    # Data for logistique training ( Load the iris dataset from sklearn )
    X,y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    
    def __init__(self, typeInt, hypers_1):
        self.lrs_epochs = hypers_1
        self.optimizerType = typeInt
        self.instances = []
               
    def plot_loss(self):
        match self.optimizerType :
            case 0:
                # Models Training
                print("0 - Gradient descent: ")
                for (lr, epochs) in self.lrs_epochs:
                    instance = Linear_regression(lr, epochs) 
                    self.instances.append(instance)
                
                for instance in self.instances:
                    instance.train_gd(self.xtrain, self.ytrain)    
                   
            case 1:
                print("1 - Stochastic Gradient descent : ")
                for (lr, epochs) in self.lrs_epochs:
                    instance = SGD(lr, epochs)
                    self.instances.append(instance)
                 
                for instance in self.instances:   
                    instance.train_with_sgd(self.xtrain, self.ytrain, 2)
                
            case 2:
                print("2 - Stochastic Gradient descent with momentum : ")
                for (lr, epochs) in self.lrs_epochs:
                    instance = SgdMomentum(lr, epochs)
                    self.instances.append(instance)
                
                for instance in self.instances:    
                    instance.train_sgd_with_momentum (X=self.xtrain, y=self.ytrain, beta=0.99)
                
            case 3:
                print("3 - Adam Gradient descent : ")
                for (lr, epochs) in self.lrs_epochs:
                    instance = AdamGD(lr, epochs)
                    self.instances.append(instance)
                
                for instance in self.instances:    
                    instance.adam_gradient_descent(self.xtrain, self.ytrain) 
                    
            case 4:
                print("4 - Mini batch Gradient descent : ")
                for (lr, epochs) in self.lrs_epochs:
                    instance = MiniBatchGD(lr, epochs)
                    self.instances.append(instance)
                    
                for instance in self.instances:    
                    instance.train_minibatch_gd(self.xtrain, self.ytrain) 
                    
            case 5:
                print("5 - Logistic regression : ")
                for (lr, epochs) in self.lrs_epochs:
                    instance = Logistic_Regression (lr, epochs )
                    self.instances.append(instance)
                
                for instance in self.instances:    
                    instance.fit(self.X_train, self.y_train)
                
        number_plot = len(self.instances)
        # Assuming we have two lines
        num_cols = number_plot // 2 if number_plot % 2 == 0 else number_plot // 2 + 1
        fig, axes = plt.subplots(2, num_cols , figsize=(15, 8))
        fig.suptitle('Training Loss')
        for i, instance in enumerate(self.instances):

            row = i // num_cols
            col = i % num_cols

            # print("row, col", row, col)
            axes[row, col].plot(instance.train_losses)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].set_title('lr = {} , n_epoch = {}'.format(instance.lr, instance.n_epochs), fontstyle='oblique')
            
        plt.tight_layout()
        plt.show()
        
                     
    
def main():
    
    
    """
    ____Prepared data for train model 
    - Gradient descent, 
    - stochastic gradient descent, 
    - stochastic gradient descent with momentum, 
    - Adam Gradient descent,
    - Minibatch gradient descent 
    """
    menu = ["- Gradient descent",  "- stochastic gradient descent,", "- stochastic gradient descent with momentum,", "- Adam Gradient descent,", "- Minibatch gradient descent", "- Logistic regression " ]
    
    print("_________ Hello ___________", end="\n")
    [print(i, menu[i], end="\n") for i in range(len(menu))]
    
    choice = int(input("choose what you want to run: "))
    while choice > len(menu):
        choice = int(input(f"choose what you want to run from 0 to: {len(menu) - 1}"))
    
    
    # Initialization of the class that manages my plots
    match choice:
        case 5:
            # For logistic regression
            linear_model = PlotHandler(choice, [(0.1, 5), (0.1, 100), (0.1, 2000), (0.001, 5000), (0.01, 5000), (0.01, 10000)])
            linear_model.plot_loss()
            return 
        
        case _ :
            # If we choose linear_regression
            linear_model = PlotHandler(choice, [(0.1, 5), (0.1, 10), (0.1, 20), (0.2, 5), (0.05, 60), (0.1, 60)])
            linear_model.plot_loss()

    
if __name__ == "__main__":
    main()
    
    
