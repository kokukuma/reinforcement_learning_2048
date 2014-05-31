#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from lib.multilayer_neural_network. multilayer_perceptron import  MultiLayerNeuralNetwork

class QLearning():
    def __init__(self, input_number, output_number):
        self.greedy_rate   = 0.2
        self.alpha         = 0.8
        self.ganmma        = 0.8
        self.input_number  = input_number
        self.output_number = output_number
        #self.q_mat = numpy.ones((input_number, output_number))
        self.mlnn = MultiLayerNeuralNetwork( [input_number, 10, output_number],
                                            threshold=0.1,
                                            start_learning_coef=0.5,
                                            sigmoid_alpha=10,
                                            mini_batch=5,
                                            output_function_type=1,
                                            epoch_limit=500)

    def predict_next(self, input_list, greedy=False):

        if greedy and self.greedy_rate < numpy.random.random():
            return numpy.random.randint(4, size=1)[0], 0
        else:
            output_vec= self.get_q_values(input_list)
            return list(output_vec).index(max(output_vec)), max(output_vec)

    def get_q_values(self, input_list):
        input_mat = numpy.array(input_list)
        #output_vec = self.mlnn.predict( input_mat.ravel() )
        output_vec = self.mlnn.predict( self.normalize_input(input_mat.ravel()))
        return output_vec

    def train(self, train_data):

        # ニューラルネットワークのトレーニングデータの形に変換
        input_data , output_data = self.change_format(train_data)
        # print "=================== INPUT"
        # for i in input_data:
        #     print i
        # print "=================== OUTPUT"
        # for i in output_data:
        #     print i
        # import sys
        # sys.exit()

        # ニューラルネットワークの学習
        error_hist = self.mlnn.train_multi(input_data , output_data)

        return error_hist

    def change_format(self, train_data):
        train_data_input  = []
        train_data_output = []
        for data in train_data:
            # 行動前の状態におけるQ値
            q_vec = self.get_q_values(data['grid'])

            # 行動後の状態におけるQ値
            move, q_value = self.predict_next(data['agrid'])

            # 特定の行動のQ値を更新する.
            # NNの出力がsoftmax関数なので, 合計が1になるように正規化.
            q_vec[data['action']]  += self.alpha * (data['point'] + self.ganmma * q_value )
            q_vec = [q/sum(q_vec) for q in q_vec]
            # print data['action'], q_vec
            # print "   ", data['point']
            # print "   ", q_value

            #train_data_input.append(numpy.array(data['grid']).ravel())
            input_data = numpy.array(data['grid']).ravel()

            train_data_input.append(self.normalize_input(input_data))
            train_data_output.append(numpy.array(q_vec))

        return numpy.array(train_data_input) , numpy.array(train_data_output)

    def normalize_input(self, train_data):
        # 入力はダミー変数化する.
        l = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        result = numpy.zeros(self.input_number)
        for i, d in enumerate(train_data):
            result[l.index(d) + i * 14 ] += 1
        return result




