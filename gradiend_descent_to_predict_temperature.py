# import section
import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression



data = pd.read_csv("./dataset/clean_weather.csv", index_col=0)
# print(data.shape)
PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"


split_data = np.split(data, [int(.7 * len(data)), int(.85 * len(data))])

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in
                                                            split_data]

def init_params(predictors):
    
    # k is a scaling factor that we use to reduce the weights and biases init
    k = math.sqrt(1 / predictors)
    
    np.random.seed(0)
    # matrix of random values scaled and centered around zero
    weights = np.random.rand(predictors, 1) * 2 * k - k
    biases = np.ones((1, 1)) * 2 * k - k
    
    return [weights, biases] 

# generate prediction in forward 
def forward(params, x):
    weights, biasses = params
    prediction = x @ weights + biasses
    # print(f"inside forward {params}")
    return prediction

# compute mse
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

# derive mse
def mse_grad(actual, predicted):
    return  predicted - actual

# To update weights and biasses
def backward(params, x, lr, grad):
    w_grad = (x.T / x.shape[0]) @ grad
    b_grad = np.mean(grad, axis = 0)
    # print(f"weight {w_grad} biasses {b_grad}")
    params[0] -= w_grad * lr
    params[1] -= b_grad * lr

    return params

lr = 1e-4
epochs = 50000
params = init_params(train_x.shape[1])

# for visualization
sample_rate = 100
samples = int (epochs / sample_rate)
historical_ws = np.zeros((samples, train_x.shape[1]))
historical_gradient = np.zeros((samples,))

for i in range(epochs):
    if i == 0:
        print(f"initial one {params}")
    predictions = forward(params, train_x)
    grad = mse_grad(train_y, predictions)
    
    params = backward(params, train_x, lr, grad)
    
    if i % sample_rate == 0:
        index = int(i / sample_rate )
        historical_gradient[index] = np.mean(grad)
        historical_ws[index,:] = params[0][:,0]
        
    if i % 10000 == 0:
        predictions = forward(params, valid_x)
        valid_loss = mse(valid_y, predictions)
        
        print(f"Eporch {i} validation loss: {valid_loss}")



