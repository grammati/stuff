import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def plotfn(f, lo=-5, hi=5):
    pts = np.linspace(lo, hi, num=100)
    sns.lineplot(x=pts, y=list(map(f, pts)))


def class_boundaries(X, Y, model):
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    pts = np.array([[x, y] for x in x_range for y in y_range])
    C = model.predict(pts)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(pts[:, 0], pts[:, 1], hue=C, palette='rainbow')
    sns.scatterplot(X[:, 0], X[:, 1], hue=Y)
