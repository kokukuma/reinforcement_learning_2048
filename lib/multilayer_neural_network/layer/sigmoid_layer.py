# -*- coding: utf-8 -*-

import numpy

class SigmoidLayer(object):
    def __init__(self):
        pass
    def _safeexp(self, input_vec):
        """入力値の上限を設定"""
        return numpy.exp(numpy.clip(input_vec, -500, 500))

    def forword(self, input_vec):
        alpha = 1.0
        return 1 / (1 + self._safeexp(- input_vec * alpha))

    def backword(self, output_vec, predict_vec):
        """誤差のnet値に置ける偏微分"""
        # pybrainの実装と違うのが気になる.
        # lib/pybrain/structure/modules/sigmoidlayer.py
        delta_vec = -1 * (output_vec - predict_vec) * predict_vec * ( 1 -  predict_vec)
        return numpy.array(delta_vec)

    def middle_update(self, next_weight_mat, next_deleta_vec, predict_vec):
        """誤差のnet値に置ける偏微分"""
        delta_mat = numpy.dot(next_deleta_vec , next_weight_mat ) * predict_vec * (1 - predict_vec)
        return delta_mat
