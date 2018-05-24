import pandas as pd
import numpy as np

class NeuralNetwork:
    '''
    A simple neural network with a single hidden layer of arbitrary size.
    '''
    def __init__(self, hidden_layer_size = 3):
        self.hidden_layer_size = hidden_layer_size
        self.parameters = None

    def structure(self, X, Y, size):
        n_x = X.shape[0]  # Number of features per training example
        n_y = Y.shape[0]  # Size of the outpit layer
        n_h = size  # Size of the hidden layer
        return n_x, n_y, n_h

    def initialize(self, n_x, n_y, n_h):
        W1 = np.random.rand(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.rand(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def tanh(self, z):
        return np.tanh(z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = self.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        _all = {'Z1': Z1,
                'A1': A1,
                'Z2': Z2,
                'A2': A2,
                }
        return A2, _all

    def calculate_cost(self, A2, Y):
        m = Y.shape[1]  # number of example
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
        cost = - np.sum(logprobs) / m
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        # Required parameters
        m = X.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']

        # Calculate the gradients
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - A1 ** 2)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def update_parameters(self, parameters, grads, learning_rate=1.2):
        # parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Get the gradients
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        # Update the coefficients
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        # New parameters
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def fit(self, X, Y, num_epochs=2000, print_output=True):

        n_h = self.hidden_layer_size

        # Neural network structure
        n_x, n_y, n_h = self.structure(X, Y, n_h)

        # Initialize parameters
        parameters = self.initialize(n_x, n_y, n_h)
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # Loop (gradient descent)

        for epoch in range(0, num_epochs):

            # Forward propagation
            A2, cache = self.forward_propagation(X, parameters)

            # Compute the cost
            cost = self.calculate_cost(A2, Y)

            # Backpropagation
            grads = self.backward_propagation(parameters, cache, X, Y)

            # Gradient descent parameter update
            parameters = self.update_parameters(parameters, grads)

            # Print the cost every 1000 iterations
            if print_output and epoch % 100 == 0:
                print("Cost after iteration %i: %f" % (epoch, cost))

        self.parameters = parameters


    def predict(self, X):
        A2, cache = self.forward_propagation(X, self.parameters)
        predictions = np.where(A2 > 0.5, 1, 0)
        return predictions

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, f1_score
    #Load some data
    df = pd.read_csv('pima-indians-diabetes.data.txt', header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    X_train = X_train.as_matrix().T
    X_test = X_test.as_matrix().T
    y_train = np.array(y_train).reshape(1, -1)
    y_test = np.array(y_test).reshape(1, -1)

    #Fit the neural network
    nn = NeuralNetwork()
    nn.fit(X_train, y_train)

    #Assess the accuracy
    predictions = nn.predict(X_test)
    print('accuracy_score: {}'.format(accuracy_score(list(predictions[0]), list(y_test[0]))))
    print('f1_score: {}'.format(f1_score(list(predictions[0]), list(y_test[0]))))





