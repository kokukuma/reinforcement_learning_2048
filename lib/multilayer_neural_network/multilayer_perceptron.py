#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
import sys
import numpy as np
from output_function import OutputFunciton
from back_propagation_logic import BackPropagationLogic
from get_training_data import liner_training_data,quadratic_function_data, sin_function_data, change_format
from numpy import linalg as LA

class MultiLayerNeuralNetwork(OutputFunciton, BackPropagationLogic):
    """
    output_function_type
        0 : logistic sigmoid function
        1 : softmax function
    """
    def __init__( self,
                  layer_construction,
                  threshold=0.5,
                  start_learning_coef=0.5,
                  sigmoid_alpha=10,
                  print_error=True,
                  mini_batch=100,
                  output_function_type=0,
                  epoch_limit=5000):

        def _two_layer_extraction(layer_construction):
            for idx, neuron_num in enumerate(layer_construction):
                if idx < len(layer_construction) - 1:
                    yield idx+1, neuron_num, layer_construction[idx+1]

        self.total_layer_num      = len(layer_construction)
        self.layer_construction   = layer_construction
        self.threshold            = threshold
        self.weights              = {}
        self.start_learning_coef  = start_learning_coef
        self.learning_coefficient = start_learning_coef
        self.error_border         = 0.01
        self.sigmoid_alpha        = sigmoid_alpha
        self.output_function_type = output_function_type
        self.print_error          = print_error
        self.mini_batch           = mini_batch
        self.epoch_limit          = epoch_limit


        # init weight
        for idx, input_num, output_num in _two_layer_extraction(layer_construction):
            weight_mat = np.ones((output_num, input_num+1))
            #weight_mat = np.zeros((output_num, input_num+1))
            for i in range(output_num):
                for j in range(input_num+1):
                    #weight_mat[i, j] *= random.uniform(-1, 1)
                    weight_mat[i, j] *= np.random.normal(0, 0.01, 1)
            weight_mat[:, 0] = -1 * self.threshold
            self.weights[idx] = weight_mat


        # # exp_table
        # self.exp_table = []
        # self.scale_factor = 1000
        # self.max_exp = 6
        # #print self.scale_factor * 2 * self.max_exp
        # for i in range(self.scale_factor * 2 * self.max_exp):
        #     # print i
        #     # print (i - self.max_exp * self.scale_factor ) / self.scale_factor
        #     exp = np.exp((i - self.max_exp * self.scale_factor ) / self.scale_factor )
        #     self.exp_table.append(exp)


        # # sigmoid_table
        # self.sigmoid_table = []
        # self.max_x = 6
        # for i in range(self.scale_factor * 2 * self.max_x):
        #     x = (i - self.max_x * self.scale_factor ) / self.scale_factor
        #     res = 1 / (1 + np.e ** x)
        #     self.sigmoid_table.append(res)




    #@profile
    def train_multi(self, train_data_input, train_data_output):
        import more_itertools

        loop_num = 0
        np_rng_input  = np.random.RandomState(1234)
        np_rng_output = np.random.RandomState(1234)

        error_hist = []

        # print
        # print 'START TRAINING'
        while True:
            error_list = []
            loop_num  += 1

            # # ループ毎に学習係数を減少させる.
            # self.learning_coefficient = self.start_learning_coef * (1 - (float(loop_num)/self.epoch_limit) ** 2)
            # self.learning_coefficient = self.learning_coefficient if self.start_learning_coef * 0.001 < self.learning_coefficient else self.start_learning_coef * 0.001

            # # シードが同じなので, 同じ形にばらされる.
            # # あってもなくても. 収束の仕方はかわるが..
            np_rng_input.shuffle(train_data_input)
            np_rng_output.shuffle(train_data_output)

            # # バックプロパゲーションで重みを更新. ミニバッチ法?
            # input_data  = train_data_input[0:self.mini_batch]
            # output_data = train_data_output[0:self.mini_batch]
            # self.weights = self.update_weights(input_data, output_data, self.weights)

            # バックプロパゲーションで重みを更新.
            start_num = 0
            for x in xrange(0, len(train_data_input)+self.mini_batch, self.mini_batch):
            #for x in xrange(0, len(train_data_input)/10, self.mini_batch):
                if x == 0:
                    continue
                input_data  = train_data_input[start_num:x]
                output_data = train_data_output[start_num:x]
                self.weights = self.update_weights(input_data, output_data, self.weights)
                start_num = x


            # print self.weights

            # 全教師データとの誤差が設定値以下になっていること.
            start_num = 0
            for x in xrange(0, len(train_data_input)+self.mini_batch, self.mini_batch):
                if x == 0:
                    continue
                #print start_num, x
                input_data  = train_data_input[start_num:x]
                output_data = train_data_output[start_num:x]
                predict_data = self.predict_multi(input_data)

                error_list.append(self.get_e_value(output_data, predict_data) )
                #print error_list
                start_num = x

            if reduce(lambda x,y : x and y , [ e < self.error_border for e in error_list]):
                if self.print_error:
                    print self.print_error
                    print 'error_list : ', error_list
                break

            # 誤差表示
            import sys
            if self.print_error:
                error = round(sum(error_list) / len(error_list),5)
                error_hist.append((loop_num, error))
                #print 'l : ', loop_num, error, [ e < self.error_border for e in error_list].count(False)
                error = '\rloop_num:%d , error:%f, error_node:%s, learn_coef:%s' % (loop_num, error, [ e < self.error_border for e in error_list].count(False), str(self.learning_coefficient) )
                sys.stdout.write(error)
                #sys.stdout.write("\r%s" % error)
                sys.stdout.flush()

            # 長過ぎたらあきらめる.
            if loop_num>self.epoch_limit:
                if self.print_error:
                    print 'out of limit'
                break
        return error_hist

    #@profile
    def update_weights(self, input_data, output_data, weights):

        predict_data_list = self.predict_all_layer(input_data)
        # print "------------------------- predict_data_list"
        # print predict_data_list[:10]

        # バックプロパゲーション
        for layer_num in reversed(range(self.total_layer_num)):

            import copy
            predict_data           = copy.deepcopy(predict_data_list[layer_num] )
            #predict_data           = copy.copy(predict_data_list[layer_num] )
            #predict_data           = predict_data_list[layer_num]

            if layer_num >= 1:
                pre_layer_predict_data = copy.deepcopy(predict_data_list[layer_num - 1])
                #pre_layer_predict_data = copy.copy(predict_data_list[layer_num - 1])
                #pre_layer_predict_data = predict_data_list[layer_num - 1]
            else:
                pre_layer_predict_data = copy.deepcopy(input_data)
                #pre_layer_predict_data = copy.copy(input_data)
                #pre_layer_predict_data = input_data

            weight_mat, output_vec, predict_vec, pre_layer_predict_vec, next_layer_weight_mat  = self.get_vecs(output_data, layer_num, weights, predict_data, pre_layer_predict_data)


            # L1-norm
            #l = 0.01
            #L1_norm = l * LA.norm(weight_mat)
            #L2_norm = l * LA.norm(weight_mat, 2)
            L1_norm = 0.
            moment  = 0.


            #print "=========="
            if layer_num == 0:
                # 入力層は重みなしのため, 更新もない.
                continue

            elif layer_num == self.total_layer_num-1:
                # 出力層の重みを更新
                #print output_vec, predict_vec
                if self.output_function_type == 2:
                    # 出力層の関数を線形にした場合.
                    #deleta_mat = self.delta_for_output_layer_linear(output_vec, predict_vec)
                    deleta_mat = self.delta_for_output_layer(output_vec, predict_vec)
                else:
                    deleta_mat = self.delta_for_output_layer(output_vec, predict_vec)
                #print deleta_mat

                # print '---------------'
                # print deleta_mat.T.shape
                # print pre_layer_predict_vec.shape
                # print np.dot(deleta_mat.T, pre_layer_predict_vec).shape

                weight_mat = weight_mat - self.learning_coefficient * np.dot(deleta_mat.T , pre_layer_predict_vec ) / self.mini_batch
                weights[layer_num] = weight_mat


                # 中間層の重み更新で利用する.
                next_deleta_vec = deleta_mat

            else:
                # 中間層の重みを更新
                #predict_vec      = np.insert(predict_vec,[0],[1.0])
                predict_vec      = np.insert(predict_vec,0,1.0,axis=1)

                # if self.output_function_type == 2:
                #     deleta_mat = self.delta_for_middle_layer_linear(next_layer_weight_mat, next_deleta_vec)
                #     deleta_mat = deleta_mat[:,1:]
                # else:
                #     deleta_mat = self.delta_for_middle_layer(next_layer_weight_mat, next_deleta_vec, predict_vec)
                #     deleta_mat = deleta_mat[:,1:]
                deleta_mat = self.delta_for_middle_layer(next_layer_weight_mat, next_deleta_vec, predict_vec)
                deleta_mat = deleta_mat[:,1:]

                weight_mat = moment * weight_mat + weight_mat - self.learning_coefficient * np.dot(deleta_mat.T, pre_layer_predict_vec )  / self.mini_batch  + L1_norm
                weights[layer_num] = weight_mat

                # 次の中間層の重み更新で利用する.
                next_deleta_vec = deleta_mat

        return weights

    def predict(self, input_data):
        predict_data_list = self.predict_all_layer(np.array([input_data]))
        result = predict_data_list[self.total_layer_num-1]
        return result[0]

    def predict_multi(self, input_data):
        predict_data_list = self.predict_all_layer(input_data)
        result = predict_data_list[self.total_layer_num-1]
        return result

    #@profile
    def predict_all_layer(self, input_data):

        output_list = [input_data]

        input_vec = np.array(input_data)

        for idx in range(1, self.total_layer_num ):
            weight_mat = self.weights[idx]

            # input_vec[0]は, 閾値用なので常に1
            #input_vec = np.insert(input_vec, [0], [1.0])
            input_vec = np.insert(input_vec, 0, 1.0, axis=1)

            net_value_vec = np.dot(weight_mat, input_vec.T).T
            if idx == self.total_layer_num - 1 and  self.output_function_type == 2:
                #output_vec = net_value_vec
                output_vec = self.sigmoid_function_tmp(net_value_vec, limit=1)

            elif idx == self.total_layer_num - 1 and self.output_function_type == 1:
                output_vec = self.softmax_function(net_value_vec)

            else:
                #output_vec = self.sigmoid_function(net_value_vec, self.sigmoid_alpha, sigmoid_table=self.sigmoid_table, scale_factor=self.scale_factor, max_x=self.max_x)
                output_vec = self.sigmoid_function(net_value_vec, self.sigmoid_alpha)

            input_vec = output_vec

            output_list.append(output_vec)

        return  output_list


def test_multilayer_perceptron():

    def plot(fig, data):
        ax  = fig.add_subplot(111)
        ax.plot([x[0] for x in data], [x[1] for x in data])

    def scat(fig, liner_data, marker='o', color='g'):
        ax  = fig.add_subplot(111)
        ax.scatter([x[0] for x in liner_data], [x[1] for x in liner_data], marker=marker, color=color, s=10)

    def get_predict_list(x_range, y_range, nn, split=10):
        data = []
        xspan = float(x_range[1] - x_range[0]) / split
        yspan = float(y_range[1] - y_range[0]) / split

        for x_value in [ float(i)*xspan+x_range[0] for i in range(split)]:
            predict_list = []
            for y_value in [ float(j) * yspan + y_range[0]  for j in range(split)]:
                if nn.predict([x_value,y_value])[0] >= 0.5:
                    data.append((x_value, y_value))
                    break
        return data

    import matplotlib.pyplot as plt
    mlnn = MultiLayerNeuralNetwork( [2, 1],
                                    threshold=0.1,
                                    start_learning_coef=0.5,
                                    sigmoid_alpha=10,
                                    mini_batch=10,
                                    output_function_type=2)

    x_range = [0,1]
    y_range = [0,1]
    #liner_data = liner_training_data(x_range, y_range)
    #liner_data = quadratic_function_data(x_range, y_range, split=20)
    liner_data = sin_function_data(x_range, y_range, 20)
    train_data_input, train_data_output = change_format(liner_data)

    fig = plt.figure()
    scat(fig, [key for key, value in liner_data.items() if value == 0], color='g' )
    scat(fig, [key for key, value in liner_data.items() if value == 1], color='b' )

    # 学習
    error_hist = mlnn.train_multi(train_data_input, train_data_output)

    # xに対応するyを算出, 学習後分離線書く
    data = get_predict_list(x_range,y_range, mlnn, split=20)
    plot(fig, data)

    # エラー表示
    fig2 = plt.figure()
    plot(fig2, error_hist)

    # 表示
    plt.show()


if __name__ == '__main__':
    test_multilayer_perceptron()
