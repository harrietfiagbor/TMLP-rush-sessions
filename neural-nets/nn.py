import numpy as np


class NeuralNet():

    def __init__(self):
        # seed the random number generator, so it generates the same numbers
        np.random.seed(1)

        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.weights = 2 * np.random.random((3, 1)) - 1

    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            # passing the training set through our neural network
            output = self.think(inputs)

            # calculating error
            error = outputs - output

            # multiplying error by input and again by gradient of sigmoid curve
            # this means less confident weights are adjusted more
            # this means inputs, which are zero, do not cause changes to the weights
            adjustments = np.dot(
                inputs.T, error * self.sigmoid_derivative(output))

            # adjusting the weights
            self.weights += adjustments

    def sigmoid(self, x):
        # activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # derivative of the Sigmoid function
        return x * (1 - x)

    def think(self, inputs):
        # passing the inputs via the neural network to get output
        inputs = inputs.astype(float)
        # passing through our neural network (our single neuron)
        # y = x * w + b
        output = self.sigmoid(np.dot(inputs, self.weights))
        return output
