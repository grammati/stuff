import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]
  delta = 1.0

  # initialize the gradient as zero
  dW = np.zeros((D,C))

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  for im, correct_class in zip(X, y):
    # scores: array of C stores, each being a float where a higher number means
    # we predict a higher probability that the image of of that class
    scores = im.dot(W) # (D,) . (D,C) == (C,)

    for j in xrange(num_classes):
      # SVM loss is the sum of incorrect class scores that exceed the correct
      # class' score by more than some margin delta. Because of that, we can
      # skip ahead when j is the correct class of the current image.
      if j == correct_class:
        continue

      margin = scores[j] - scores[correct_class] + delta
      if margin > 0:
        loss += margin

        # The following line is very hard to explain. It has to do with the
        # gradient of the hinge-loss, which takes some fancy math to derive.
        # Essentially it is saying this: Because the score in class j (which is
        # not the correct class for this image) exceeds the score for the
        # correct class by more than delta, the parameters in the j-th column
        # of W contributed too much to scores[j] for this image. We need the
        # parameters in column j to be smaller. How much smaller? Let's
        # illustrate with a specific example. Let's look at when j is 7. Then
        # the parameter at, say, W[3,7] (the parameter that decides how much
        # the value of pixel number 3 contributes to making the image look like
        # it's in class 7) contributed W[3,7] * im[3] to scores[7]. Similarly
        # for other indices (other than 3), up to D (the dimension of the
        # inputs). So the gradient on column j of W due to this input is just
        # `im`. To see this, consider a single cell in W - the example above of
        # W[3,7]. The contribution to the loss is W[3,7] * im[3]. Considering
        # W[3,7] as the variable, and im[3] as a constant, the derivative of
        # that product with respect to W[3,7] is just im[3]. So we need to add
        # im[3] to the value of dW[3,7]. Similarly for indices other than 3. So
        # we just add im to column j of dW.
        dW[:, j] += im

        # Now consider that calculating the score for this (incorrect) class,
        # j, involves subtracting the score of the correct class, which is the
        # dot product of column `c` of W with `im` (this is just due to how the
        # hinge-loss is defined). So for every incorrect class j, the values of
        # the parameters in the correct-class column of W have a negative
        # contribution to scores[j].
        dW[:, correct_class] -= im

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Ditto for the gradient
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Contribution to the gradient due to the regularization. I note that the 2
  # from taking the derivative of W**2 cancels out with the 0.5 above (which is
  # exactly the reason for including that 0.5 in the first place.)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # Better names for the dimensions:
  N = X.shape[0]
  D = X.shape[1]
  C = y.shape[0]

  scores = X.dot(W) # (N,D) . (D,C) == (N,C)
  correct_class_scores = scores[np.arange(N), y] # (N,)
  margins = scores - correct_class_scores.reshape((-1,1)) + 1 # (N,C)
  # Correct for overcounting when score == correct (i.e. correct for the "+ 1" above)
  margins[np.arange(N), y] = 0
  loss = np.sum(np.maximum(margins, 0)) / N

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # For each of the N samples X[i], margin_counts[i] will be a count of the
  # number of classes whose score contributes to the loss.
  margin_counts = np.sum(margins > 0, axis=1) # (N,)

  # The following lines of code are far and away the most difficult I have ever
  # written. I have spent days figuring them out, and filled pages and pages of
  # a notebook with diagrams, and I'm still not sure I really understand why
  # this works. Somewhat unsatisfyingly, the answer is "because the dimensions
  # work out". That seems to be how a lot my numpy work is done - if the
  # dimensions match up, then I assume it's right. I really hope this stuff
  # eventually gets easier.
  xcounts = (margins > 0).astype('int') # why?  (N,C)
  xcounts[np.arange(N), y] -= margin_counts  # this one I think I understand
  dW = X.T.dot(xcounts)  # (D,N) . (N,C) == (D,C)
  dW /= N

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
