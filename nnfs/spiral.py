# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'nnfs'))
    print(os.getcwd())
except:
    pass

# %%
from nnfs.layers import FC, Relu, Softmax
from nnfs.core import NN
from nnfs.datasets import spiral
from nnfs import plotting
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from IPython import get_ipython
ip = get_ipython()
ip.magic('load_ext autoreload')
ip.magic('autoreload 2')

# %%
X, Y = spiral(3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette='Spectral')

# %%
nn = NN(
    FC(2, 20),
    Relu(),
    FC(20, 20),
    Relu(),
    FC(20, 3),
    Softmax()
)

# %%
for _ in range(10):
    for _ in range(100):
        loss = nn.train(X, Y)
    print(nn.accuracy(X, Y), loss)

# %%
plt.plot(loss)

# %%
plotting.class_boundaries(X, Y, nn)

#%%
