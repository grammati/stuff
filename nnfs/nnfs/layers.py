import numpy as np


class Layer(object):
    def loss(self):
        return 0


class FC(Layer):
    'Fully connected layer'

    def __init__(self, d_in, d_out):
        self.d_in = d_in
        self.d_out = d_out
        self.W = np.random.randn(d_in, d_out) / 100
        self.b = np.zeros(d_out)

    def forward(self, x):
        # x @ W + b
        self.dW = x.T
        return (x @ self.W) + self.b

    def backward(self, grad_in):
        dW = self.dW @ grad_in
        dW += 0.001 * self.W
        db = grad_in.sum(axis=0)
        db += 0.001 * self.b
        self.W -= 0.1 * dW
        self.b -= 0.1 * db
        return grad_in @ self.W.T

    def loss(self):
        return 0.005 * (self.W * self.W).sum()


class Relu(Layer):
    def forward(self, x):
        self.grads = x >= 0
        return x * self.grads

    def backward(self, grad_in):
        return grad_in * self.grads


class Softmax(Layer):
    def forward(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(x)
        probs = exps / exps.sum(axis=-1, keepdims=True)
        self.probs = probs
        return probs

    def backward(self, grad_in):
        return grad_in
