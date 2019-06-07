# # %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
# import os
# try:
#     os.chdir(os.path.join(os.getcwd(), 'nnfs'))
#     print(os.getcwd())
# except:
#     pass

# %%
from nnfs.core import sigmoid
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from nnfs.datasets import moons
from nnfs.core import NN
from nnfs.layers import FC, Relu, Softmax, Layer
import random

# %%
from IPython import get_ipython
ip = get_ipython()
ip.magic('load_ext autoreload')
ip.magic('autoreload 2')

# %%


def relu(x):
    x = x.copy()
    x[x < 0] = 0
    return x


class RNN:
    """Char-RNN"""

    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        V = self.vocab_size
        H = self.hidden_size
        Y = self.vocab_size

        self.h = np.zeros(H)
        self.Wh = np.random.randn(H, H)
        self.bh = np.zeros(H)

        self.Wx = np.random.randn(V, H)
        self.bx = np.random.randn(H)

        self.Wy = np.random.randn(H, Y)
        self.by = np.zeros(Y)

    def step(self, x, y):
        '''x and y are characters. A step does the following:
          - predict the probabilities of each character that could follow `x`. This will
            be a discrete probability distribution of cardinality `vocab_size` (in other
            words, an array of `vocab_size` floats that sum to 1).
          - these probabilities are calculated as follows:
            - calculate a new `h` (hidden state) based on both the current hidden state
              and the value of `x`, as `W_x @ x + W_h @ h + b_h`
            - caculate the class-scores as `W_y @ new_h + b_y`
        '''
        V = self.vocab_size
        H = self.hidden_size
        Y = self.vocab_size

        # one-hot inputs
        assert x.shape == (V,)
        assert y.shape == (Y,)

        # Forward
        x_out = x @ self.Wx + self.bx
        h_out = self.h @ self.Wh + self.bh
        new_h = relu(x_out) + relu(h_out)
        assert new_h.shape == (H,)

        y_hat = new_h @ self.Wy + self.by  # (H,) @ (H,Y) = (Y,)
        assert y_hat.shape == (Y,)

        # calculate probability for each output class (character)
        # TODO: watch out for overflow
        probs = np.exp(y_hat) / np.sum(np.exp(y_hat))

        loss = np.sum(-np.log(probs[y] + 1e-8))

        # Backward - update Wh, Wx, and Wy
        grad = probs - y

        dby = grad
        assert dby.shape == self.by.shape
        dWy = np.outer(new_h.T, grad)  # (H,) * (Y,) = (H,Y)
        assert dWy.shape == self.Wy.shape

        dnew_h = self.Wy @ grad  # (H,Y) @ (Y,) = (H,)
        assert dnew_h.shape == (H,), f'shape: {dnew_h.shape}'

        dbh = dnew_h
        assert dbh.shape == self.bh.shape
        dWh = np.outer(self.h, dnew_h)  # (H,) * (H,) = (H,H)
        dWh[:, h_out < 0] = 0  # relu
        assert dWh.shape == self.Wh.shape

        dbx = dnew_h
        assert dbx.shape == self.bx.shape
        dWx = np.outer(x, dnew_h)  # (V,) * (H,) = (V,H)
        dWx[:, x_out < 0] = 0  # relu
        assert dWx.shape == self.Wx.shape

        # Gradient Descent
        lr = 3e-3
        self.Wx += lr * dWx
        self.bx += lr * dbx
        self.Wh += lr * dWh
        self.bh += lr * dbh
        self.Wy += lr * dWy
        self.by += lr * dby

        self.h = new_h

        return loss

    def learn(self, text):
        chars = sorted(set(text))
        vocab = {c: i for i, c in enumerate(chars)}
        V = len(chars)
        x = np.zeros(V, dtype=int)
        y = np.zeros(V, dtype=int)
        losses = []
        for c1, c2 in zip(text, text[1:]):
            x[:] = 0
            x[vocab[c1]] = 1
            y[:] = 0
            y[vocab[c2]] = 1
            loss = self.step(x, y)
            losses.append(loss)
        self.vocab = vocab

    def predict(self, n):
      c = random.choice(self.vocab.keys())
      for i in range(n):
        probs = 

# %%
def test():
    rnn = RNN(3, 5)
    t = 'abc' * 1000
    rnn.learn(t)


# %%
test()


# %%
