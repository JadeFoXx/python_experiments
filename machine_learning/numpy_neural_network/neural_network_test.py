from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork
from neural_network import sigmoid
from neural_network import relu
from neural_network import bin_cross_entropy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
sns.set_style("whitegrid")

FILE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = 'plots'

GRID_X_START = -1.5
GRID_X_END = 2.5
GRID_Y_START = -1.0
GRID_Y_END = 2
grid = np.mgrid[GRID_X_START:GRID_X_END:100j,GRID_X_START:GRID_Y_END:100j]
grid_2d = grid.reshape(2, -1).T
XX, YY = grid

def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()

X, y = make_moons(1000, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

nn = NeuralNetwork([2,25,50,50,25,1], activation_function=relu, readout_function=sigmoid, cost_function=bin_cross_entropy, seed=2)

def callback_numpy_plot(index):
    plot_title = "NumPy Model - It: {:05}".format(index)
    file_name = "numpy_model_{:05}.png".format(index//50)
    file_path = os.path.join(FILE_DIR, OUTPUT_DIR)
    file_path = os.path.join(file_path, file_name)
    prediction_probs, _ = nn.forward_prop(np.transpose(grid_2d))
    prediction_probs = prediction_probs[-1]
    prediction_probs = prediction_probs.reshape(prediction_probs.shape[1], 1)
    make_plot(X_test, y_test, plot_title, file_name=file_path, XX=XX, YY=YY, preds=prediction_probs, dark=True)

nn.train(X_train.T, np.transpose(y_train.reshape((y_train.shape[0], 1))), 10000, 0.01, verbose=True, callback=callback_numpy_plot)


