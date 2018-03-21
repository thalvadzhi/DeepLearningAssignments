import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        n_nonnegative_terms = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                n_nonnegative_terms += 1
        # gradient with respect to w_yi is -xi multiplied by the number of
        # non-zero terms in the sum for the loss function
        dW[:, y[i]] -= X[i] * n_nonnegative_terms

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # average the gradient too
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # since every weight in W * W is the square of a weight in W
    # a 2 comes out in front
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    W = np.float64(W)
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X @ W
    true_values = scores[np.arange(len(scores)), y]
    deltas = np.maximum(
        0, scores - true_values.reshape(num_train, 1) + 1)

    # respect the j =/= yi in the sum, otherwise you have to subtract 1 at the end
    deltas[np.arange(len(scores)), y] = 0
    loss = np.sum(deltas) / num_train
    # if instead I multiply by 2 in the gradient I get an overflow
    loss += 0.5 * reg * np.sum(W * W)

    n_nonnegative = np.zeros(deltas.shape)
    n_nonnegative[np.arange(num_train), y] = np.sum(deltas > 0, axis=1)

    dW += X.T @ (deltas > 0)
    dW -= X.T @ n_nonnegative
    dW = dW / num_train
    dW += reg * W
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
    # pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
