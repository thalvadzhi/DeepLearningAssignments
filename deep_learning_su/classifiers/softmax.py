import numpy as np
from random import shuffle
from past.builtins import xrange


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

    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
        scores = X[i] @ W
        # handle numerical instability
        scores -= np.max(scores)
        correct_class_score = scores[y[i]]
        exps = np.exp(scores)
        p = np.exp(correct_class_score) / np.sum(exps)
        loss += -np.log(p)

        for cl in range(num_classes):
            if cl != y[i]:
                dW[:, cl] += (exps[cl] / np.sum(exps)) * X[i]
        dW[:, y[i]] -= (1 - p) * X[i]

        ##############
        # derivative of p is
        # -p * (exps[cl] / np.sum(exps)) * X[i] if i != cl
        # p * (1 - p) * X[i] if i == cl
        # derivative of -log(p) = - 1/p * derivative of p
        # so der of -log(p) =
        # (exps[cl] / np.sum(exps)) * X[i] if i != cl
        # - (1 - p) * X[i] if i == cl

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X @ W
    # handle numerical instability
    scores -= np.max(scores, axis=1).reshape((num_train, 1))
    correct_class_scores = scores[np.arange(num_train), y]
    correct_class_exps = np.exp(correct_class_scores)
    sum_row = np.sum(np.exp(scores), axis=1)
    ps = correct_class_exps / sum_row

    loss = np.mean(-np.log(ps))
    loss += reg * np.sum(W * W)

    scores_exp = np.exp(scores)
    summed_scores_exp_by_row = np.sum(
        scores_exp, axis=1).reshape((num_train, 1))
    scores_exp /= summed_scores_exp_by_row
    # scores_exp is a matrix with num_train rows and num_classes columns
    # each element is the score for that example and that class exponentiated and divided 
    # by the sum of exps for that example.

    # set the scores to zero for the correct classes for every example because
    # differentiating by the weight of the score of the corect class is done with another formula
    scores_exp[np.arange(num_train), y] = 0

    # this is equivalent to dW[:, cl] += (exps[cl] / np.sum(exps)) * X[i]
    dW += (scores_exp.T @ X).T

    one_minus_ps = (1 - ps).reshape((1, num_train))
    # marks the correct class of every example with a 1
    mark_correct_class = np.zeros((num_classes, num_train))
    mark_correct_class[y, np.arange(num_train)] = 1

    # multiply by 1 - the softmax of the scores of the correct classes
    mark_correct_class *= one_minus_ps
    
    # this is equivalent to dW[:, y[i]] -= (1 - p) * X[i]
    dW -= (mark_correct_class @ X).T

    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
