"""
Inspired in cs231n/classfiers/fc_net.py

Implement a multimodal net

"""

import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *
from cs231n.multimodal import multimodal_utils


class MultiModalNet(object):
    """

    """

    def __init__(self, img_input_dim, txt_input_dim, hidden_dim, weight_scale, reg=0.0, seed=None):
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

        # Initialize local loss hyperparams
        self.local_margin = None
        self.local_scale = None
        self.do_mil = None

        # Initialize global score hyperparams
        self.global_margin = None
        self.global_scale = None
        self.global_method = None
        self.thrglobalscore = None
        self.smooth_num = None
        self.non_lin_fun = None

        # Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.

        if seed:
            np.random.seed(seed)

        self.params['Wi2s'] = weight_scale['img'] * np.random.randn(img_input_dim, hidden_dim)
        self.params['bi2s'] = np.zeros(hidden_dim)
        self.params['Wsem'] = weight_scale['txt'] * np.random.randn(txt_input_dim, hidden_dim)
        self.params['bsem'] = np.zeros(hidden_dim)

    def set_global_score_hyperparams(self, global_margin=40., global_scale=1., smooth_num=5.,
                                     global_method='maxaccum', thrglobalscore=False):
        self.global_margin = global_margin
        self.global_scale = global_scale
        self.global_method = global_method
        self.thrglobalscore = thrglobalscore
        self.smooth_num = smooth_num

    def set_local_hyperparams(self, local_margin, local_scale, do_mil):
        self.local_margin = local_margin
        self.local_scale = local_scale
        self.do_mil = do_mil

    def loss_local(self, sim_region_word, y):
        """

        """
        # verify that local hyper params have been set
        assert self.local_margin is not None, "you need to set local_margin"
        assert self.local_scale is not None, "you need to set local_scale"
        assert self.do_mil is not None, "specify whether you want to do multiple instance learning"

        local_margin = self.local_margin
        local_scale = self.local_scale
        do_mil = self.do_mil

        loss, d_local_scores = svm_two_classes(sim_region_word, y, delta=local_margin, do_mil=do_mil)

        return loss * local_scale, d_local_scores * local_scale

    def loss_global(self, sim_region_word, region2pair_id, word2pair_id):

        # verify that global hyper params have been set
        assert self.global_margin is not None, "set global margin"
        assert self.global_scale is not None, "set global scale"
        assert self.global_method is not None, "set global method, either sum or maxaccum"
        assert self.thrglobalscore is not None, "set whether you want to threshold scores at 0 ??"
        assert self.smooth_num is not None, "set smoothing constant for global score"

        global_margin = self.global_margin
        global_scale = self.global_scale
        global_method = self.global_method
        thrglobalscore = self.thrglobalscore
        smooth_num = self.smooth_num

        # numbers of pairs in batch
        N = np.max(region2pair_id) + 1
        assert N == np.max(word2pair_id) + 1

        y = np.arange(N)  # the correct pairs correspond to the diagonal elements

        img_sent_score_global, SGN, img_region_with_max = global_scores_forward(sim_region_word, N, region2pair_id,
                                                                                word2pair_id, smooth_num,
                                                                                thrglobalscore=thrglobalscore,
                                                                                global_method=global_method)

        loss, d_global_scores = svm_struct_loss(img_sent_score_global, y, delta=global_margin, avg=False)

        d_local_scores = global_scores_backward(d_global_scores, N, sim_region_word,
                                                region2pair_id, word2pair_id, SGN,
                                                img_region_with_max,
                                                global_method=global_method, thrglobalscore=thrglobalscore)

        return loss * global_scale, d_local_scores * global_scale

    def loss(self, X_img, X_txt, region2pair_id, word2pair_id, **kwargs):
        """
        Compute loss and gradient for a minibatch of data.

        Before calling this method, the loss function must be set

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

        # unnpack keyword arguments
        uselocal = kwargs.pop('uselocal', False)
        useglobal = kwargs.pop('useglobal', False)

        assert uselocal or useglobal, "at least one of them must be set to True"

        # initialize cost
        loss = 0
        grads = {}

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
        if region2pair_id is None and word2pair_id is None:
            return sim_region_word

        dscores = np.zeros((sim_region_word.shape))

        if uselocal:
            # make an appropriate y
            y = multimodal_utils.pair_id2y(region2pair_id, word2pair_id)
            local_loss, dscores1 = self.loss_local(sim_region_word, y)
            loss += local_loss
            dscores += dscores1

        if useglobal:
            global_loss, dscores2 = self.loss_global(sim_region_word, region2pair_id, word2pair_id)
            loss += global_loss
            dscores += dscores2

        reg_loss = 0.5 * self.reg * (np.sum(Wi2s * Wi2s) +
                                     np.sum(Wsem * Wsem))

        loss += reg_loss  # add the regularization loss.

        # ############################################################################
        # # Implement the backward pass for the multimodal net.
        # # Store gradients in the grads dictionary.
        # # Make sure that grads[k] holds the gradients for  #
        # # self.params[k]. Don't forget to add L2 regularization!                   #
        # ############################################################################

        d_proj_imgs, d_proj_txt = mult_backward(dscores, cache_mult)

        dX_img, dWi2s, dbi2s = affine_backward(d_proj_imgs, cache_proj_imgs)

        dX_txt, dWsem, dbsem = affine_relu_backward(d_proj_txt.T, cache_proj_txt)

        # d_proj_txt is allDeltasSent in matlab
        # d_proj_imgs is allDeltasImg

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

    # def loss_all(self, X, y, delta=1, do_mil=False, normalize=True):
    #     """
    #     Compute loss and gradient for a minibatch of data.
    #
    #     Inputs:
    #     - X: Array of input data of shape (n_regions_in_batch, n_words_in_batch)
    #     - y: Array of labels same shape as scores. y[i,j] gives the label for X[i,j].
    #
    #     Returns:
    #     If y is None, return none since we don't need to compute the cost
    #
    #     If y is not None, then run a training-time forward and backward pass and
    #     return a tuple of:
    #     - loss: Scalar value giving the loss
    #     - grads: Dictionary with the same keys as self.params, mapping parameter
    #       names to gradients of the loss with respect to those parameters.
    #     """
    #     loss, dX = svm_two_classes(X, y, delta, do_mil, normalize)





