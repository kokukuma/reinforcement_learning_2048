#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math

class BackPropagationLogic(object):

    def update_weights(self, input_data, output_vec, weights):
        """ weigthsを１回更新
        """
        # 出力層以外, biosのinput付き.
        predict_data_list = self.predict_all_layer(input_data)

        # backprop
        for layer_num in reversed(range(self.total_layer_num)):
            predict_vec = predict_data_list[layer_num]

            if layer_num == 0:
                # 入力層は重みなしのため, 更新もない.
                continue
            elif layer_num == self.total_layer_num-1:
                # 出力層
                deleta_mat = self.layer_type[layer_num].backword(output_vec, predict_vec)
            else:
                # 中間層
                deleta_mat  = self.layer_type[layer_num].middle_update(weights[layer_num+1], next_deleta_mat, predict_vec)
                deleta_mat  = deleta_mat[:,1:]

            pre_layer_predict_vec = predict_data_list[layer_num - 1]
            weights[layer_num]    = self.gradientdescent_list[layer_num](weights[layer_num], deleta_mat, pre_layer_predict_vec)

            # 中間層の重み更新で利用する.
            next_deleta_mat = deleta_mat

        return weights

    #@profile
    def predict_all_layer(self, input_vec):
        """ 各層のoutputを返却. biosの入力付き.
        """
        input_vec   = self.add_bios_input(input_vec)
        output_list = [input_vec]
        for idx in range(1, self.total_layer_num ):
            # input_vec[0]は, 閾値用なので常に1
            #input_vec = self.add_bios_input(input_vec)

            net_value_vec = np.dot(self.weights[idx] , input_vec.T).T
            output_vec    = self.layer_type[idx].forword(net_value_vec)

            if not idx == self.total_layer_num - 1:
                output_vec = self.add_bios_input(output_vec)
                input_vec  = output_vec
            output_list.append(output_vec)
        return  output_list

    def add_bios_input(self, input_vec):
        return  np.insert(input_vec,0,1.0,axis=1)


