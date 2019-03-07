import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from decoder.main import normal
from mpl_toolkits.mplot3d import Axes3D


def load_data(file_name):
    mat_dict = loadmat(file_name, appendmat=True)
    return np.array(mat_dict['Tte']), np.array(mat_dict['Ttr']), np.array(mat_dict['Xte']), np.array(mat_dict['Xtr']), np.array(mat_dict['Yte']), np.array(mat_dict['Ytr'])


def color_data(X, id):
    id = np.reshape(np.array(id), len(id))
    id_size = np.amax(id)
    data_size = X.shape[1]

    for i in range(id_size + 1):
        plt.plot(X[id == i, 0], X[id == i, 1], linestyle='', marker='.', markersize=15)


def color_data_3D(X, id):
    id = np.reshape(np.array(id), len(id))
    id_size = np.amax(id)
    data_size = X.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(id_size + 1):
        ax.scatter(X[id == i, 0], X[id == i, 1], X[id == i, 2], marker='.')


def eval_R2(X, Y):
    X = normal(X)
    Y = normal(Y)
    return 1 - np.mean(np.power(Y - X, 2), axis=0).sum() / np.var(Y, axis=0).sum()
