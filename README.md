# Implementation in python of regression (Linear and logistic :😜)

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





















