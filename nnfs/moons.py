# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'nnfs'))
    print(os.getcwd())
except:
    pass

# %%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from nnfs.datasets import moons
from nnfs.core import NN
from nnfs.layers import FC, Relu, Softmax
# %%
from IPython import get_ipython
ip = get_ipython()
ip.magic('load_ext autoreload')
ip.magic('autoreload 2')


# %%
X, Y = moons()
sns.scatterplot(X[:, 0], X[:, 1], hue=Y)
nn = NN(
    FC(2, 20),
    Relu(),
    FC(20, 20),
    Relu(),
    FC(20, 2),
    Softmax())


# %%
losses = []
fig, ax = plt.subplots()
for i in range(100):
    for _ in range(100):
        losses.append(nn.train(X, Y))
    if not ax.lines:
        print('Plotting')
        plt.plot(losses)
    else:
        ax.lines[0].set_xdata(np.arange(len(losses)))
        ax.lines[0].set_ydata(losses)
    fig.canvas.draw()
print(f'Done: {i}')

# %%
xrange = np.linspace(-1.5, 2.5, 100)
yrange = np.linspace(-1.5, 1.5, 100)
pts = np.array([[x, y] for x in xrange for y in yrange])
print(pts.shape)
C = nn.predict(pts)
print(C.shape)
plt.figure(figsize=(10, 6))
sns.scatterplot(pts[:, 0], pts[:, 1], hue=C, palette='Accent')
sns.scatterplot(X[:, 0], X[:, 1], hue=Y)


# %%
plt.matshow(nn.layers[4].W)


# %%
