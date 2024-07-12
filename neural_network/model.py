import math

import numpy as np

from neural_network.layers import DropoutLayer


class Model:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.training = False

    def predict(self, inputs):
        predicted = inputs
        for layer in self.layers:
            if not self.training and isinstance(layer, DropoutLayer):
                continue
            layer.forward(predicted)
            predicted = layer.output
        return predicted

    def __learn(self, batched_inputs, learning_rate):
        grad = self.loss.grad
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].backward(self.layers[i - 1].output, grad, learning_rate)
            grad = self.layers[i].grad
        self.layers[0].backward(batched_inputs, grad, learning_rate)

    def __fill_validation_history(self, validation_data, history):
        self.training = False
        validation_inputs, actual_validation_outputs = validation_data
        predicted_validation_outputs = self.predict(validation_inputs)
        self.loss.forward(predicted_validation_outputs, actual_validation_outputs)
        validation_loss = self.loss.output
        history["validation_loss"].append(validation_loss)
        tp_count = np.sum(np.argmax(predicted_validation_outputs, axis=1) == actual_validation_outputs)
        all_count = len(actual_validation_outputs)
        validation_accuracy = tp_count / all_count
        history["validation_accuracy"].append(validation_accuracy)
        self.training = True

    def train(self, inputs, outputs, epochs, batch_size, decay, validation_data=None):
        history = {
            "training_loss": [],
            "validation_loss": [],
            "validation_accuracy": []
        }

        self.training = True
        idx = list(range(len(inputs)))
        for epoch in range(epochs):
            epoch_losses = []
            np.random.shuffle(idx)
            shuffled_inputs = inputs[idx]
            shuffled_outputs = outputs[idx]
            current_learning_rate = decay.calculate_new_learning_rate(epoch)
            for start in range(0, len(shuffled_inputs), batch_size):
                end = min(start + batch_size, len(shuffled_inputs))
                batched_inputs = shuffled_inputs[start:end]
                actual_outputs = shuffled_outputs[start:end]
                predicted_outputs = self.predict(batched_inputs)
                self.loss.forward(predicted_outputs, actual_outputs)
                epoch_losses.append(self.loss.output)
                self.loss.backward(predicted_outputs, actual_outputs)
                self.__learn(batched_inputs, current_learning_rate)

            training_loss = np.mean(epoch_losses)
            history["training_loss"].append(training_loss)
            if validation_data is not None:
                self.__fill_validation_history(validation_data, history)

            print(f"{epoch}: {training_loss}")
        self.training = False
        return history
