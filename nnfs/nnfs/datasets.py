from sklearn import datasets as ds
import numpy as np


def moons():
    x1, y1 = ds.make_moons(n_samples=100, shuffle=True, noise=0.2)
    x2, y2 = ds.make_moons(n_samples=100, shuffle=True, noise=0.2)
    x2 += [2, 1.2]
    y2 += 2
    X = np.concatenate([x1, x2])
    Y = np.concatenate([y1, y2])
    # return X, Y
    return x1, y1


def spiral(n_classes: int = 3,
           points_per_class: int = 100,
           inner_radius: float = 0.0,
           fuzziness: float = 0.2):
    D = 2  # dimensionality
    N = points_per_class
    K = n_classes
    X = np.zeros((N*K, D))  # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(inner_radius, 1.0, N)  # radius
        t = np.linspace(j*6.28/K, (j+2)*6.28/K, N) + \
            np.random.randn(N) * fuzziness  # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y
