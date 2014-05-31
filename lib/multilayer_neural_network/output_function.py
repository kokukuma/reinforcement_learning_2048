#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
#import pytest

class OutputFunciton(object):

    @classmethod
    def step_function(self, net_value_vec, bottom=0):
        def _step(x_value):
            return 1 if x_value>=0 else bottom
        result =  [ _step(value) for value in net_value_vec]
        return np.array(result)

    # @classmethod
    # @profile
    # def softmax_function2(self, net_value_vec):
    #     e = np.exp(net_value_vec - np.max(net_value_vec) )
    #     if e.ndim == 1:
    #         return e / np.sum(e, axis=0)
    #     else:
    #         #return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2
    #         return e / np.sum(e, axis=1).reshape(e.shape[0] , 1)  # ndim = 2

    @classmethod
    #@profile
    def softmax_function(self, net_value_vec):
        e = np.exp(net_value_vec)  # prevent overflow
        npsum = np.sum(e, axis=1)
        npres = npsum.reshape(e.shape[0] , 1)
        return e / npres

    @classmethod
    def sigmoid_function(self, net_value_vec, alpha=5):
        # ## vectorize
        # def _sigmoid(x_value, alpha):
        #     if x_value * alpha < -5:
        #         return 1
        #     elif x_value * alpha > 5:
        #         return 0
        #     else:
        #         return 1 / ( 1 + np.e ** (-1 * x_value * alpha))
        # vfunc = np.vectorize(_sigmoid)
        # return vfunc(net_value_vec, alpha)

        ### 直接. ほんとに合ってるのか?
        # return 1 / (1 + np.e ** (-1 * net_value_vec * alpha))
        return 1 / (1 + np.exp(- net_value_vec * alpha))         ## 一番正しいやつ.

        ### exp_tableをつかう.
        # def _select_exp_table(x):
        #     if x < - max_x:
        #         return 0
        #     elif x > max_x:
        #         return 1
        #     else:
        #         i = int((x+max_x) * scale_factor)
        #         return sigmoid_table[i]

        # vfunc = np.vectorize(_select_exp_table)
        # return vfunc( - net_value_vec * alpha)

        ### sigmoidの代わりに、tanhを使う.... もう...これは...
        #return np.tanh(net_value_vec)



    @classmethod
    def probability_sigmoid_function(self, net_value_vec, alpha=5):
        # = 1 / ( 1 + e^-x)
        def _sigmoid(x_value):
            prob =  1 / ( 1 + math.e ** (-1 * x_value * alpha))
            rnd = random.uniform(0, 1)
            return 1 if rnd < prob else 0

        result =  [ _sigmoid(value) for value in net_value_vec]
        return np.array(result)


    @classmethod
    def function_apply(self, net_value_vec, output_function_type):
        if output_function_type == 0:
            result = self.step_function(net_value_vec)
        elif output_function_type == 1:
            result = self.sigmoid_function(net_value_vec)
        return np.array(result)
