import numpy as np


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, derivatives):
        self.grad = derivatives * (self.output > 0)


class LeakyReLU:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, inputs):
        self.output = np.maximum(self.alpha * inputs, inputs)

    def backward(self, derivatives):
        self.grad = derivatives * ((self.output > 0) + (self.output <= 0) * self.alpha)


class Softmax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, derivatives):
        self.grad = np.zeros(self.output.shape)
        for i in range(len(self.output)):
            mat = np.zeros((self.output.shape[1], self.output.shape[1]))
            for j in range(mat.shape[0]):
                for k in range(mat.shape[1]):
                    if j == k:
                        mat[j, k] = self.output[i, j] - self.output[i, j] ** 2
                    else:
                        mat[j, k] = -1 * self.output[i, j] * self.output[i, k]
            self.grad[i] = np.dot(derivatives[i], mat)


class Sigmoid:
    def forward(self, inputs):
        inputs_clipped = np.clip(inputs, -500, 500)
        self.output = 1 / (1 + np.exp(-inputs_clipped))

    def backward(self, derivatives):
        self.grad = derivatives * self.output * (1 + self.output)

