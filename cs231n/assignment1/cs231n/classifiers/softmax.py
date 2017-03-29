import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  assert X.shape == (N,D)
  assert W.shape == (D,C)
  assert y.shape == (N,)

  for n,X in enumerate(X):
    actual_class = y[n]
    scores = X.dot(W)
    assert scores.shape == (C,)

    # Avoid INF
    scaled_scores = scores - np.max(scores)
    exp_scores = np.exp(scaled_scores)
    sigma = np.sum(exp_scores)
    probs = exp_scores / sigma

    loss -= np.log(probs[actual_class])

    for c in range(C):
        if c == actual_class:
            dW[:,c] += (probs[c] - 1.0) * X
        else:
            dW[:,c] += (probs[c]) * X

  loss /= N
  dW /= N

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  scores = X.dot(W) # (N,D) . (D,C) = (N,C)
  scores -= np.max(scores, axis=1, keepdims=True)
  scores = np.exp(scores)
  sigma = np.sum(scores, axis=1, keepdims=1)
  probs = scores / sigma
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N

  probs[np.arange(N), y] -= 1.0  # (N,C)
  dW = X.T.dot(probs) / N        # (D,N) . (N,C) = (D,C)

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

