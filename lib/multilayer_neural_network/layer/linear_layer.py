# -*- coding: utf-8 -*-

import numpy

class LinearLayer(object):
    def __init__(self):
        pass
    def forword(self, input_vec):
        return input_vec

    def backword(self, output_vec, predict_vec):
        """誤差のnet値に置ける偏微分"""
        # pybrainの実装だと, そのままyを返してる.
        # lib/pybrain/structure/modules/linearlayer.py
        delta_vec = -1 * (output_vec - predict_vec)
        return numpy.array(delta_vec)

    def middle_update(self, next_weight_mat, next_deleta_vec, predict_vec):
        """誤差のnet値に置ける偏微分"""
        delta_mat = numpy.dot(next_deleta_vec , next_weight_mat ) * predict_vec * (1 - predict_vec)
        return delta_mat
