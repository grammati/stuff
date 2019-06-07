import numpy as np


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


class NN:
    def __init__(self, *layers):
        self.layers = layers

    def train(self, X, Y):
        n_examples = len(X)

        v = X
        for layer in self.layers:
            v = layer.forward(v)

        probs = v
        eps = 1e-8
        cel = -np.log(probs[np.arange(n_examples), Y] + eps)
        L = cel.sum() / n_examples

        z = np.zeros_like(probs)
        z[np.arange(n_examples), Y] += 1
        grad = (probs - z) / n_examples
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return L

    def predict(self, x):
        v = x
        for layer in self.layers:
            v = layer.forward(v)
        return np.argmax(v, axis=-1)

    def accuracy(self, X, Y):
        preds = self.predict(X)
        return (preds == Y).sum() / len(preds)