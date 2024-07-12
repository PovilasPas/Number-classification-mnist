import numpy as np


class LossCategoricalCrossEntropy:
    def forward(self, predicted, actual):
        size = len(predicted)
        predicted_clipped = np.clip(predicted, 1e-6, 1 - 1e-6)
        if len(actual.shape) == 1:
            probs = predicted_clipped[range(size), actual]
        else:
            probs = np.sum(predicted_clipped * actual, axis=1)
        loss = -1 * np.log(probs)
        self.output = np.mean(loss)

    def backward(self, predicted, actual):
        predicted_clipped = np.clip(predicted, 1e-6, 1 - 1e-6)
        if len(actual.shape) == 1:
            hot_encoded = np.zeros(predicted_clipped.shape)
            for i in range(hot_encoded.shape[0]):
                for j in range(hot_encoded.shape[1]):
                    if actual[i] == j:
                        hot_encoded[i, j] = 1
            actual = hot_encoded
        self.grad = -1 * actual / predicted_clipped





