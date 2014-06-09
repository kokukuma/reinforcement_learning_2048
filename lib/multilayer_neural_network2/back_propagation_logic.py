#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

class BackPropagationLogic(object):

    @classmethod
    #@profile
    def get_e_value(self, output_data, predict_data):

        if not len(output_data) == len(predict_data):
            return None
        return np.sum((predict_data - output_data ) ** 2 ) / 2 / len(predict_data)

    @classmethod
    def delta_for_output_layer_linear(self, output_vec, predict_vec):
        # 出力層が線形ダッタ場合の更新.
        delta_mat    = -1 * (output_vec - predict_vec)
        return np.array(delta_mat)

    @classmethod
    def delta_for_middle_layer_linear(self, next_weight_mat, next_deleta_vec):
        # 中間層が線形ダッタ場合の更新.
        delta_mat = np.dot(next_deleta_vec , next_weight_mat )
        return delta_mat


    @classmethod
    def delta_for_output_layer(self, output_vec, predict_vec):
        # 誤差のnet値における偏微分
        delta_mat    = -1 * (output_vec - predict_vec) * predict_vec * ( 1 -  predict_vec)
        return np.array(delta_mat)

    @classmethod
    def delta_for_middle_layer(self, next_weight_mat, next_deleta_vec, predict_vec):
        # print next_weight_mat.shape
        # print next_deleta_vec.shape
        # print predict_vec.shape
        #print np.dot(next_deleta_vec , next_weight_mat )
        # 誤差のnet値における偏微分
        delta_mat = np.dot(next_deleta_vec , next_weight_mat ) * predict_vec * (1 - predict_vec)
        return delta_mat
