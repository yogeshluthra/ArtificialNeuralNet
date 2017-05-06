import numpy as np

"""
Plain Feed Forward Neural Network
The chosen activation function is the Leaky ReLU function
Courtesy: https://github.com/FlankMe/general-gym-player/blob/master/GeneralGymPlayerWithNP.py
@author: Riccardo Rossi

Additions by Yogesh Luthra: RMSProp implementation in _backprop
"""
class FeedForwardNeuralNetwork:
    def __init__(self, layers, ALPHA=2.5e-4, learnMomentum=0.9):
        # ---NN learn settings
        self._ALPHA = ALPHA
        self.learnMomentum = learnMomentum

        # NN variables
        self._generateNetwork(np.array(layers))

    def _generateNetwork(self, layers):
        """
        The network is implemented in Numpy
        Change this method if you wish to use a different library
        """

        # Initialization parameters
        INIT_WEIGHT_STD = 0.01
        INITIALIZATION_BIAS = 0.01

        # Activation function used is the Leaky ReLU function
        self._activation = lambda x: x * (0.01 * (x < 0) + (x >= 0))
        self._derive = lambda x: 0.01 * (x < 0) + (x >= 0)

        # Create the graph's architecture
        self._weights = []
        self.Eg_W_sq = []       # expected weight gradient (but most recent experiences weighted more than past grads)
        self._bias = []
        self.Eg_B_sq = []       # expected bias gradient (but most recent experiences weighted more than past grads)
        self.eTol=1e-8          # error tolerance

        for i in range(layers.size - 1):
            self.Eg_W_sq.append(np.zeros(shape=(layers[i], layers[i + 1]), dtype=float))
            self._weights.append(np.random.normal(scale=INIT_WEIGHT_STD,
                                       size=(layers[i], layers[i + 1])))

            self.Eg_B_sq.append(np.zeros(layers[i + 1], dtype=float))
            self._bias.append(INITIALIZATION_BIAS * np.ones((layers[i + 1])))

    def _feedFwd(self, X):

        self._activation_layers = [np.atleast_2d(X)]

        for i in range(len(self._weights) - 1):
            self._activation_layers.append(self._activation(
                np.dot(self._activation_layers[-1],
                       self._weights[i]) +
                self._bias[i]))

        # Last layer does not require the activation function
        self._activation_layers.append(
            np.dot(self._activation_layers[-1],
                   self._weights[-1]) +
            self._bias[-1])

        return (self._activation_layers[-1])

    def _backProp(self, X, a, y):

        # Calculate the delta vectors
        self._delta_layers = [a * (np.atleast_2d(y).T - self._feedFwd(X))]

        #---Don't need to compute gradient for very first layer (as it is just input X)
        for i in range(len(self._activation_layers) - 2, 0, -1):
            self._delta_layers.append(
                    np.dot(self._delta_layers[-1],
                           self._weights[i].T) *
                    self._derive(self._activation_layers[i]))
        #------------Number of gradient layers = Number of activation layers-1 = Num of weights

        self._delta_layers.reverse()

        # Update the weights and bias vectors, using RMSProp algorithm (see http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
        for i in range(len(self._weights)):
            g_W = np.dot(self._activation_layers[i].T, self._delta_layers[i])   # find weight gradient
            self.Eg_W_sq[i] = self.learnMomentum * self.Eg_W_sq[i] \
                              + (1. - self.learnMomentum) * g_W**2 # RMSProp expected weight update

            g_B = self._delta_layers[i].sum(axis=0) # find bias gradient
            self.Eg_B_sq[i] = self.learnMomentum * self.Eg_B_sq[i] \
                              + (1. - self.learnMomentum) * g_B**2 # RMSProp expected bias update

            self._weights[i]    += self._ALPHA * (g_W / np.sqrt(self.Eg_W_sq[i] + self.eTol))
            self._bias[i]       += self._ALPHA * (g_B / np.sqrt(self.Eg_B_sq[i] + self.eTol))

    def predict(self, state):
        return (self._feedFwd(state))

    def fit(self, valueStates, actions, valueTarget):
        self._backProp(valueStates, actions, valueTarget)


