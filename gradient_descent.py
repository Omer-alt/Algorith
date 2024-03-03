# simple example to learn descent gradient
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2 + y**2

def g(x, a):
    return x * a

def df_dx (x):
    return 2 * x

def df_dy (y):
    return 2 * y

def gradient_descent (x_start, y_start, learnig_rate, num_iterations):

    x = x_start
    y = y_start
    history = []

    for i in range (num_iterations):
        print("progression of the parameters of the line xa+y=f(x,y) \n")
        print ("iteration \n", (x, y, f(x,y)))
        grad_x = df_dx (x)
        grad_y = df_dy(y)

        x = x - learnig_rate * grad_x
        y = y - learnig_rate * grad_y
        
        history.append((x, y, f(x,y)))

    return  x, y, f(x,y), history

# Define the meshgrid for plotting the function
x_range = np.arange(-10, 10, 0.1)
y_range = np.arange(-10, 10, 0.1)
# Create the X coordinate grid in the direction of x and Y in that from y
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# perform gradient descent and plot the result
start_x, start_y = 8, 8
learning_rate = 0.1
num_iterations = 20
x_opt, y_opt, f_opt, history = gradient_descent (start_x, start_y, learning_rate, num_iterations )

fig = plt.figure()
# computed_zorder : for ordering manualy intersecting curves
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.scatter(*zip(*history), c='r', marker='o')

# Plot the trajectory of gradient descent on the surface 
ax.scatter(*zip(*history), c='r', marker='o')

# Plot the curve x * a + y (best fit of error function in green color)
a = 0.5  # Coefficient to optimize
curve_x = np.linspace(-10, 10, 100)
curve_y = g(curve_x, a)
ax.plot(curve_x, curve_y, f(curve_x, curve_y), c='g')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

plt.show()





