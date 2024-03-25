# Implementation in python of Machine learning basic algorithm : Linear and logistic gradient descent, PCA, Neural network etc 😜

This repository presents the basics of machine learning, particularly regression.

## I. Linear regression

the following graphs show the results of minimizing the error function following a parameter variation

### 1. Gradient descent
![gradient_descent](/public/assets/gradient_descent.png)
### 2. Stochiastic Gradient descent
![stochastic_gradient_descent](/public/assets/stochastic_gradient_descent.png)
### 3. Stochiastic Gradient descent with momentum
Considering a fixed beta to calculate momentum  (equal to 0.99)
![gradient_descent_with_momentum](/public/assets/gradient_descent_with_momentum.png)

### 4. Minibatch Gradient descent 
Considering a fixed batch (equal to 3)
![minibatch_gradient_descent](/public/assets/minibatch_gradient_descent.png)
### 5. Adam Gradient descent
Considering fixed variables like (beta1=0.9, beta2=0.999, epsilon=1e-8) 
![adam_gradient_descent](/public/assets/adam_gradient_descent.png)
## II. Logistic regression

![logistic_gradient_descent](/public/assets/logistic_gradient_descent.png)

## Optimizations

What about optimizations in this code ? You can notice the usage of
-  OOP paradigm
- The single responsibility principle
E.g. refactors, performance improvements, accessibility
## III. Neural network for Classification
## 1 - Problem to solve
In this section it is a question of carrying out the classification of data which is presented as the Xor logic gate (see the data graph below).

![xor_data_set](/public/assets/xor_data_set.png)

## 2 - Resolution approach
to solve this classification problem we propose to use a neural network with a single hidden layer as follows:

![Neural_network](/public/assets/Neural1-Page-2.png)

The activation function to use is the sigmoid function and to minimize our loss we use the gradient descent. Below are the results of our decision boundary and our loss (for training and testing sets)

**Losses:** 
![Train_test_loss](/public/assets/Losses.png)
**Decision Boundary:** 
![Train_test_loss](/public/assets/decision_boundary.png)
## Tech Stack

**Language:** Python

**Package:** Numpy, Sklearn, matplotlib, pandas, ipywidgets

## Run Locally

Clone the project

```bash
  git clone https://github.com/Omer-alt/Basic_ML_Algorithm.git
```

Go to the project directory

```bash
  cd my-project
```

Run the main file

```bash
  main.py
```



## Authors

- [@Fotso omer](https://portfolio-omer-alt.vercel.app/)

## License

[MIT](https://choosealicense.com/licenses/mit/)





















