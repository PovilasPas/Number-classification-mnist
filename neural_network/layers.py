import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons, fn, initializer):
        self.weights = initializer.generate_weights(n_inputs, n_neurons)
        self.biases = np.ones((1, n_neurons)) * 0.1
        self.fn = fn

    def forward(self, inputs):
        result = np.dot(inputs, self.weights) + self.biases
        self.fn.forward(result)
        self.output = self.fn.output

    def backward(self, inputs, derivatives, lr):
        self.fn.backward(derivatives)
        self.grad = np.dot(self.fn.grad, self.weights.T)

        dldw = np.dot(inputs.T, self.fn.grad)
        lenw = np.linalg.norm(dldw, axis=0) + 1e-6
        dldw /= lenw
        self.weights -= lr * dldw

        dldb = np.sum(self.fn.grad, axis=0)
        lenb = np.linalg.norm(dldb) + 1e-6
        dldb /= lenb
        self.biases -= lr * dldb


class DropoutLayer:
    def __init__(self, p_keep):
        self.p_keep = p_keep

    def forward(self, inputs):
        self.mask = np.random.binomial(1, self.p_keep, size=inputs.shape) / self.p_keep
        self.output = inputs * self.mask

    def backward(self, inputs, derivatives, lr):
        self.grad = derivatives * self.mask








