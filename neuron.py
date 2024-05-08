import numpy as np

def sigmoid(x):
    # activation function
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # derivative of sigmoid
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # weight inputs, add bias, then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class NeuralNetwork:
    '''
        - 2 inputs
        - hidden layer with 2 neurons
        - output layer with 1 neuron
    '''
    def __init__(self) -> None:
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1

    def train(self, data, all_y_trues):
        '''
            - data is a (n x 2) numpy array, n = # of samples
            - all_y_trues is a numpy array with n elements
            - elements in all_y_trues correspond to those in data
        '''

    learn_rate = 0.1
    epochs = 1000

network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))



y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))
