import math

import numpy as np


class HeNormal:
    @staticmethod
    def generate_weights(n_inputs, n_neurons):
        std = math.sqrt(2.0/n_inputs)
        output = np.random.randn(n_inputs, n_neurons) * std
        return output


class GlorotNormal:
    @staticmethod
    def generate_weights(n_inputs, n_neurons):
        std = math.sqrt(2.0/(n_inputs + n_neurons))
        output = np.random.randn(n_inputs, n_neurons) * std
        return output

