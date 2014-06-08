#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
import sys
import numpy as np
from output_function import OutputFunciton
from back_propagation_logic import BackPropagationLogic
from get_training_data import liner_training_data,quadratic_function_data, sin_function_data, change_format

from layer.linear_layer import LinearLayer
from layer.sigmoid_layer import SigmoidLayer
from layer.softmax_layer import SoftmaxLayer

def add_bios_input(input_vec):
    return  np.insert(input_vec,0,1.0,axis=1)

class GradientDescent(object):
    """ 傾きに対していろいろやる
    """
    def __init__(self, learning_coefficient, mini_batch, momentum=1.0, decay=1.0, regular=None):
        self.learning_coefficient = learning_coefficient
        self.mini_batch           = mini_batch
        self.momentum             = momentum
        self.decay                = decay
        self.regular              = regular
        self.l                    = 0.01
        self.norm                 = 0.

    def __call__(self, weight_mat, deleta_mat, pre_layer_predict_vec):
        """ 更新後の重みを返す
        """
        from numpy import linalg as LA
        # TODO: この正則化, 層選べないじゃん.
        if self.regular == 'L1':
            self.norm = self.l * LA.norm(weight_mat)
        elif self.regular == 'L2':
            self.norm = self.l * LA.norm(weight_mat, 2)

        grad = np.dot(deleta_mat.T , pre_layer_predict_vec ) / self.mini_batch +  self.norm

        self.learning_coefficient *= self.decay

        return self.momentum * weight_mat - self.learning_coefficient * grad


class MultiLayerNeuralNetwork(OutputFunciton, BackPropagationLogic):
    """
    """
    def __init__( self,
                  layer_construction,
                  threshold=0.5,
                  start_learning_coef=0.5,
                  sigmoid_alpha=10,
                  print_error=True,
                  mini_batch=100,
                  layer_type=[LinearLayer, SigmoidLayer, SigmoidLayer],
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
        self.error_border         = 0.01
        self.sigmoid_alpha        = sigmoid_alpha
        self.print_error          = print_error
        self.mini_batch           = mini_batch
        self.epoch_limit          = epoch_limit
        self.gradientdescent      = GradientDescent(self.start_learning_coef, self.mini_batch)

        # TODO: add tanh /gausian
        self.layer_type = [x() for x in layer_type]

        # init weight
        for idx, input_num, output_num in _two_layer_extraction(layer_construction):
            weight_mat = np.ones((output_num, input_num+1))
            for i in range(output_num):
                for j in range(input_num+1):
                    weight_mat[i, j] *= np.random.normal(0, 0.01, 1)
            weight_mat[:, 0] = -1 * self.threshold
            self.weights[idx] = weight_mat

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
                error = '\rloop_num:%d , error:%f, error_node:%s, learn_coef:%s' % (loop_num, error, [ e < self.error_border for e in error_list].count(False), str(self.start_learning_coef) )
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
            weights[layer_num]    = self.gradientdescent(weights[layer_num], deleta_mat, pre_layer_predict_vec)

            # 中間層の重み更新で利用する.
            next_deleta_mat = deleta_mat

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
    def predict_all_layer(self, input_vec):
        """ 各層のoutputを返却. biosの入力付き.
        """
        input_vec   = add_bios_input(input_vec)
        output_list = [input_vec]
        for idx in range(1, self.total_layer_num ):
            # input_vec[0]は, 閾値用なので常に1
            #input_vec = add_bios_input(input_vec)

            net_value_vec = np.dot(self.weights[idx] , input_vec.T).T
            output_vec    = self.layer_type[idx].forword(net_value_vec)

            if not idx == self.total_layer_num - 1:
                output_vec = add_bios_input(output_vec)
                input_vec  = output_vec
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
    mlnn = MultiLayerNeuralNetwork( [2, 5, 1],
                                    threshold=0.1,
                                    start_learning_coef=0.2,
                                    sigmoid_alpha=10,
                                    mini_batch=10)

    x_range = [0,1]
    y_range = [0,1]
    #liner_data = liner_training_data(x_range, y_range)
    liner_data = quadratic_function_data(x_range, y_range, split=20)
    #liner_data = sin_function_data(x_range, y_range, 20)
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
