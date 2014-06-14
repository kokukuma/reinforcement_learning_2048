#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class GradientDescent(object):
    """ 傾きに対していろいろやる
    """

    def __init__(self, learning_coefficient, mini_batch, momentum=1.0, decay=1.0, regular=None, rprop=False):
        self.learning_coefficient = learning_coefficient
        self.mini_batch           = mini_batch
        self.momentum             = momentum
        self.decay                = decay
        self.regular              = regular
        self.l                    = 0.01
        self.norm                 = 0.
        self.rprop                = rprop
        if self.rprop:
            self.etaplus  = 1.2
            self.etaminus = 0.5
            self.deltamax = 5.0
            self.deltamin = 1.0e-6
            self.lastgradient = None

    def __call__(self, weight_mat, deleta_mat, pre_layer_predict_vec):
        """ 更新後の重みを返す
        """
        grad = np.dot(deleta_mat.T , pre_layer_predict_vec ) / self.mini_batch

        if self.rprop:
            return self._rprop(weight_mat, grad)
        else:
            return self._backprop(weight_mat, grad)

    def _backprop(self, weight_mat, grad):
        """ 通常のbackprop
        """
        from numpy import linalg as LA
        # TODO: この正則化, 層選べないじゃん.
        if self.regular == 'L1':
            self.norm = self.l * LA.norm(weight_mat)
        elif self.regular == 'L2':
            self.norm = self.l * LA.norm(weight_mat, 2)

        self.learning_coefficient *= self.decay

        return self.momentum * weight_mat - self.learning_coefficient * (grad) + self.norm

    def _rprop(self, weight_mat, grad):
        """ iRprop-
        """
        if self.lastgradient == None:
            self.lastgradient = np.zeros(grad.shape, dtype='float64')
            self.rprop_theta  = self.lastgradient + 0.1

        #gradient_arr = np.asarray(grad)
        gradient_arr = np.asarray(grad)

        # update param
        delta = np.sign(gradient_arr)  * self.rprop_theta

        # update param
        dirSwitch = self.lastgradient * gradient_arr
        self.rprop_theta[dirSwitch > 0] *= self.etaplus
        self.rprop_theta[dirSwitch < 0] *= self.etaminus

        gradient_arr[dirSwitch < 0] = 0
        self.rprop_theta = self.rprop_theta.clip(min=self.deltamin, max=self.deltamax)

        # save
        self.lastgradient = gradient_arr.copy()

        return weight_mat - delta
