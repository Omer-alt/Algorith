import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

class NeuronLayer:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.W = np.random.randn(num_neurons, num_inputs)
        self.b = np.random.randn(num_neurons, 1)

    def forward_pass(self, X):
        self.Z = np.dot(self.W, X) + self.b
        self.A = Neuron().sigmoid(self.Z)
        return self.A

    def backward_pass(self, dL_dA, A_prev):
        dA_dZ = Neuron().d_sigmoid(self.Z)
        dL_dZ = dL_dA * dA_dZ
        self.dL_dW = np.dot(dL_dZ, A_prev.T)
        self.dL_db = np.sum(dL_dZ, axis=1, keepdims=True)
        self.dL_dA_prev = np.dot(self.W.T, dL_dZ)
        return self.dL_dA_prev

class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, hidden_layers):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.layers = []

        # Input layer
        self.layers.append(NeuronLayer(num_inputs, hidden_layers[0]))

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(NeuronLayer(hidden_layers[i-1], hidden_layers[i]))

        # Output layer
        self.layers.append(NeuronLayer(hidden_layers[-1], num_outputs))

    def forward_pass(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward_pass(A)
        return A

    def backward_pass(self, dL_dA):
        for i in range(len(self.layers)-1, 0, -1):
            dL_dA = self.layers[i].backward_pass(dL_dA, self.layers[i-1].A)
        return dL_dA

    def train(self, X_train, Y_train, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            A = self.forward_pass(X_train)

            # Compute loss
            loss = np.mean(np.square(Y_train - A))

            # Backward pass
            dL_dA = -(Y_train - A)
            self.backward_pass(dL_dA)

            # Update weights and biases
            for layer in self.layers:
                layer.W -= learning_rate * layer.dL_dW
                layer.b -= learning_rate * layer.dL_db

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X_test):
        return self.forward_pass(X_test)

    def accuracy(self, y_pred, y_true):
        y_pred = (y_pred > 0.5).astype(int)
        return np.mean(y_pred == y_true)

# Exemple d'utilisation
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y_train = np.array([[0, 1, 1, 0]])

model = NeuralNetwork(num_inputs=2, num_outputs=1, hidden_layers=[4, 3])
model.train(X_train, Y_train, learning_rate=0.1, epochs=1000)

y_pred = model.predict(X_train)
accuracy = model.accuracy(y_pred, Y_train)
print(f'Accuracy: {accuracy}')
