import math


class InverseTimeDecay:
    def __init__(self, initial_learning_rate, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def calculate_new_learning_rate(self, epoch_number):
        return self.initial_learning_rate / (1 + self.decay_rate * epoch_number)


class ExponentialDecay:
    def __init__(self, initial_learning_rate, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def calculate_new_learning_rate(self, epoch_number):
        return self.initial_learning_rate / math.pow(math.e, self.decay_rate * epoch_number)


class NoDecay:
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def calculate_new_learning_rate(self, _):
        return self.initial_learning_rate
