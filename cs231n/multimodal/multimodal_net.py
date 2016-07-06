"""
Implement a multimodal net
"""

import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *


class MultiModalNet(object):
    """

    """

    def __init__(self, img_input_dim, txt_input_dim, hidden_dim, weight_scale, reg=0.0):
        """
        In practice, the current recommendation is to use ReLU units
        and use the w = np.random.randn(n) * sqrt(2.0/n), as discussed in He et al..
        http://arxiv-web3.library.cornell.edu/abs/1502.01852
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

        """

        self.params = {}
        self.reg = reg
        self.h = hidden_dim
        self.loss_function = None

        # Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.

        self.params['Wi2s'] = weight_scale['img'] * np.random.randn(img_input_dim, hidden_dim)
        self.params['bi2s'] = np.zeros(hidden_dim)
        self.params['Wsem'] = weight_scale['txt'] * np.random.randn(txt_input_dim, hidden_dim)
        self.params['bsem'] = np.zeros(hidden_dim)

    def set_loss_function(self, f):
        self.loss_function = f

    def loss(self, X_img, X_txt, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        ############################################################################
        # Implement the forward pass for the multimodal net, computing the    #
        # class scores for X_img and X_txt and storing them in the sim_resion_word
        # variable.              #
        ############################################################################

        assert self.loss_function

        Wi2s = self.params['Wi2s']  # (img_input_dim, hidden_dim)
        bi2s = self.params['bi2s']  # (hidden_dim,)
        Wsem = self.params['Wsem']  # (txt_input_dim, hidden_dim)
        bsem = self.params['bsem']  # (hidden_dim,)


        # Project images into multimodal space
        projected_imgs, cache_proj_imgs = affine_forward(X_img, Wi2s, bi2s)

        # Project text into multimodal space
        projected_txt, cache_proj_txt = affine_relu_forward(X_txt, Wsem, bsem)

        # Compute the similarity between regions and words
        sim_region_word, cache_mult = mult_forward(projected_imgs, projected_txt.T)

        # If y is None then we are in test mode so just return scores
        #  (ie, similarity between regions and words)
        if y is None:
            return sim_region_word

        ############################################################################
        # Compute the cost with a svm layer (regular svm)
        ############################################################################
        data_loss, dscores = self.loss_function(sim_region_word, y)

        reg_loss = 0.5 * self.reg * (np.sum(Wi2s * Wi2s) +
                                     np.sum(Wsem * Wsem))

        loss = data_loss + reg_loss  # add the data loss and the regularization loss.

        ############################################################################
        # Implement the backward pass for the multimodal net.
        # Store gradients in the grads dictionary.
        # Make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        ############################################################################
        grads = {}

        d_proj_imgs, d_proj_txt = mult_backward(dscores, cache_mult)

        dX_img, dWi2s, dbi2s = affine_backward(d_proj_imgs, cache_proj_imgs)

        dX_txt, dWsem, dbsem = affine_relu_backward(d_proj_txt.T, cache_proj_txt)

        # add the contribution of the regularization term to the gradient
        dWi2s += self.reg * Wi2s
        dWsem += self.reg * Wsem

        # Store gradients in dictionary
        grads['Wi2s'] = dWi2s
        grads['bi2s'] = dbi2s
        # grads['X_img'] = dX_img
        grads['Wsem'] = dWsem
        grads['bsem'] = dbsem
        # grads['X_txt'] = dX_txt

        return loss, grads







