#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
import sys
import numpy as np
from evaluate_error import  EvaluateError
from dradient_descent import GradientDescent
from output_function import OutputFunciton
from back_propagation_logic import BackPropagationLogic
from get_training_data import liner_training_data,quadratic_function_data, sin_function_data, change_format

from layer.linear_layer import LinearLayer
from layer.sigmoid_layer import SigmoidLayer
from layer.softmax_layer import SoftmaxLayer



class MultiLayerNeuralNetwork(BackPropagationLogic, EvaluateError):
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
                  epoch_limit=5000,
                  rprop=False):

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
        self.gradientdescent_list = []
        self.best_weights         = {}
        self.best_error           = None

        # grad
        for i in range(self.total_layer_num):
            tmp = GradientDescent(self.start_learning_coef, self.mini_batch, rprop=rprop)
            self.gradientdescent_list.append(tmp)

        # TODO: add tanh /gausian
        self.layer_type = [x() for x in layer_type]

        # init weight
        for idx, input_num, output_num in _two_layer_extraction(layer_construction):
            weight_mat = np.ones((output_num, input_num+1))
            #weight_mat[:] *= np.random.normal(0, 0.01, 1)
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
        np_rng_input.shuffle(train_data_input)
        np_rng_output.shuffle(train_data_output)

        validate_dataset = {}
        validate_dataset['input'], train_data_input  = self.split_data(train_data_input, 0.1)
        validate_dataset['output'],train_data_output = self.split_data(train_data_output, 0.1)

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
                if x == 0:
                    continue
                input_data  = train_data_input[start_num:x]
                output_data = train_data_output[start_num:x]
                self.weights = self.update_weights(input_data, output_data, self.weights)
                start_num = x

            # トレーニングデータでエラーを確認.
            predict_data = self.predict_multi(train_data_input)
            train_error  = self.get_rss(train_data_output, predict_data)

            # 検証用データでエラーを確認.
            predict_data = self.predict_multi(validate_dataset['input'])
            valid_error  = self.get_rss(validate_dataset['output'], predict_data)

            #
            if self.best_error == None or self.best_error > train_error:
                self.best_error        = train_error
                self.best_valid_error  = valid_error
                self.best_weights = self.weights

            if train_error < self.error_border:
                break
            error_hist.append((loop_num, train_error))

            # 誤差表示
            import sys
            if self.print_error:
                train_error = '\rloop_num:%d , train_error:%f, valid_error:%s' % (loop_num, train_error, valid_error)
                sys.stdout.write(train_error)
                sys.stdout.flush()

            # 長過ぎたらあきらめる.
            if loop_num>self.epoch_limit:
                if self.print_error:
                    print 'out of limit'
                break

        print self.best_error, self.best_valid_error
        self.weights = self.best_weights

        return error_hist

    #@profile

    def predict(self, input_data):
        predict_data_list = self.predict_all_layer(np.array([input_data]))
        result = predict_data_list[self.total_layer_num-1]
        return result[0]

    def predict_multi(self, input_data):
        predict_data_list = self.predict_all_layer(input_data)
        result = predict_data_list[self.total_layer_num-1]
        return result


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
    # backpropのときは, mini_batch = 10
    # rpropのときは, mini_batch = 100
    mlnn = MultiLayerNeuralNetwork( [2, 5, 1],
                                    threshold=0.1,
                                    start_learning_coef=0.2,
                                    sigmoid_alpha=10,
                                    mini_batch=100,
                                    layer_type=[LinearLayer, SigmoidLayer, SigmoidLayer],
                                    rprop=True
                                    )

    x_range = [0,1]
    y_range = [0,1]
    #liner_data = liner_training_data(x_range, y_range)
    #liner_data = quadratic_function_data(x_range, y_range, split=20)
    liner_data = sin_function_data(x_range, y_range, split=20)
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
