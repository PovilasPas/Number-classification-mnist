import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from neural_network.utils import read_mnist, fixed_normalization

from neural_network.losses import LossCategoricalCrossEntropy
from neural_network.activation_functions import LeakyReLU, Softmax
from neural_network.initializers import HeNormal, GlorotNormal
from neural_network.decays import ExponentialDecay
from neural_network.layers import DenseLayer, DropoutLayer
from neural_network.model import Model

EPOCHS = 70
BATCH_SIZE = 2000
LR = 0.05

X, y = read_mnist(
    "data/train-images-idx3-ubyte/train-images-idx3-ubyte",
    "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)

X = fixed_normalization(X, 0, 255, 0, 1)

# TRAINING AND VALIDATION DATA SPLIT
X_train = np.array([]).reshape(0, X.shape[1])
y_train = np.array([], dtype=np.int32)
X_valid = np.array([]).reshape(0, X.shape[1])
y_valid = np.array([], dtype=np.int32)

for i in range(10):
    take = y == i
    X_filtered = X[take]
    y_filtered = y[take]
    index = round(0.9 * len(X_filtered))
    X_train = np.concatenate((X_train, X_filtered[:index]))
    y_train = np.concatenate((y_train, y_filtered[:index]))
    X_valid = np.concatenate((X_valid, X_filtered[index:]))
    y_valid = np.concatenate((y_valid, y_filtered[index:]))

# MODEL ARCHITECTURE
nn = Model([
    DenseLayer(784, 256, LeakyReLU(), HeNormal),
    DropoutLayer(0.8),
    DenseLayer(256, 128, LeakyReLU(), HeNormal),
    DropoutLayer(0.5),
    DenseLayer(128, 64, LeakyReLU(), HeNormal),
    DropoutLayer(0.5),
    DenseLayer(64, 32, LeakyReLU(), HeNormal),
    DropoutLayer(0.5),
    DenseLayer(32, 16, LeakyReLU(), HeNormal),
    DropoutLayer(0.5),
    DenseLayer(16, 10, Softmax(), GlorotNormal)
], LossCategoricalCrossEntropy())
history = nn.train(X_train, y_train, EPOCHS, BATCH_SIZE, ExponentialDecay(LR, 0.002), (X_valid, y_valid))

# PLOTS
plt.plot(np.arange(1, EPOCHS + 1), history["training_loss"], label="Training loss")
plt.plot(np.arange(1, EPOCHS + 1), history["validation_loss"], label="Validation loss")
plt.legend()
plt.title("Change in loss through epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

plt.plot(np.arange(1, EPOCHS + 1), history["validation_accuracy"])
plt.title("Change in accuracy on validation set through epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

#READ TEST DATA
X_test, y_test = read_mnist(
    "data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
    "data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
)

X_test = fixed_normalization(X_test, 0, 255, 0, 1)

y_pred = np.argmax(nn.predict(X_test), axis=1)

# CONFUSION MATRIX
cm = np.zeros((10, 10))

for actual, predicted in zip(y_test, y_pred):
    cm[actual, predicted] += 1

# ROW WISE NORMALIZED CONFUSION MATRIX
group_counts = ["{0:0.0f}".format(val) for val in cm.flatten()]
group_percentages = ["{0:0.2%}".format(val) for val in (cm/np.sum(cm, axis=1).reshape(-1, 1)).flatten()]
labels = [f"{v1}\n{v2}" for v1, v2, in zip(group_counts, group_percentages)]
labels = np.array(labels).reshape(cm.shape)
fmt = lambda x, pos: "{0:0.2%}".format(x)

sns.heatmap((cm/np.sum(cm, axis=1).reshape(-1, 1)), annot=labels, cmap="viridis", fmt="", cbar_kws={"format": FuncFormatter(fmt)})
plt.show()

# COLUMN WISE NORMALIZED CONFUSION MATRIX
group_percentages = ["{0:0.2%}".format(val) for val in (cm/np.sum(cm, axis=0)).flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.array(labels).reshape(cm.shape)

sns.heatmap((cm/np.sum(cm, axis=0)), annot=labels, cmap="viridis", fmt="", cbar_kws={"format": FuncFormatter(fmt)})
plt.show()

with open("../model.pickle", "wb") as fd:
    pickle.dump(nn, fd)
