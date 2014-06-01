#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from multilayer_neural_network. multilayer_perceptron import  MultiLayerNeuralNetwork

class QLearning():
    def __init__(self, input_number, output_number, dummy=False):
        self.l = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        self.dummy = dummy
        self.greedy_rate   = 0.2
        self.alpha         = 0.5
        self.ganmma        = 0.9
        if self.dummy:
            self.input_number  = input_number * len(self.l)
        else:
            self.input_number  = input_number
        self.output_number = output_number
        #self.q_mat = numpy.ones((input_number, output_numbe))
        self.mlnn = MultiLayerNeuralNetwork( [self.input_number,  20, self.output_number],
                                            threshold=0.5,
                                            start_learning_coef=0.1,
                                            sigmoid_alpha=10,
                                            print_error=True,
                                            mini_batch=1,
                                            output_function_type=2,
                                            epoch_limit=100)

    def predict_next(self, input_list, greedy=False):

        if greedy and self.greedy_rate < numpy.random.random():
            return numpy.random.randint(4, size=1)[0], 0
        else:
            output_vec = self.get_q_values(input_list)
            return list(output_vec).index(max(output_vec)), max(output_vec)

    def get_q_values(self, input_list):
        input_mat = numpy.array(input_list)
        #output_vec = self.mlnn.predict( input_mat.ravel() )
        output_vec = self.mlnn.predict( self.normalize_input(input_mat.ravel()))

        #return output_vec
        return map((lambda x: x if x >0 else 0), output_vec)

    def train(self, train_data):

        # ニューラルネットワークのトレーニングデータの形に変換
        input_data , output_data = self.change_format(train_data)
        #input_data , output_data = self.change_format_total_score(train_data)

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
        # for inp,  out in zip(input_data , output_data):
        #     error_hist = self.mlnn.train_multi(numpy.array([inp]),  numpy.array([out]))

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
            q_vec[data['action']]  += self.alpha * (data['point'] + self.ganmma * q_value  - q_vec[data['action']])

            #print self.alpha * (data['point'] + self.ganmma * q_value  - q_vec[data['action']])

            # NNの出力がsoftmax関数なので, 合計が1になるように正規化.
            # q_vec[data['action']]  += self.alpha * (data['point'] + self.ganmma * q_value )
            # q_vec = [q/sum(q_vec) for q in q_vec]

            # print data['action'], q_vec
            # print "   ", data['point']
            # print "   ", q_value

            #train_data_input.append(numpy.array(data['grid']).ravel())
            input_data = numpy.array(data['grid']).ravel()

            train_data_input.append(self.normalize_input(input_data))
            train_data_output.append(numpy.array(q_vec))

        return numpy.array(train_data_input) , numpy.array(train_data_output)

    def change_format_total_score(self, train_data):
        train_data_input  = []
        train_data_output = []
        q_list = []
        for data in reversed(train_data):
            # 行動前の状態におけるQ値
            q_vec = self.get_q_values(data['grid'])

            # 行動後の状態におけるQ値
            move, q_value = self.predict_next(data['agrid'])

            # 特定の行動のQ値を更新する.
            q_list = [float(q) * self.ganmma for q in q_list]
            q_list.append(data['point'])
            q_vec[data['action']]  += self.alpha * (sum(q_list) + self.ganmma * q_value  - q_vec[data['action']])
            #print sum(q_list)

            #train_data_input.append(numpy.array(data['grid']).ravel())
            input_data = numpy.array(data['grid']).ravel()

            train_data_input.append(self.normalize_input(input_data))
            train_data_output.append(numpy.array(q_vec))

        return numpy.array(train_data_input) , numpy.array(train_data_output)

    def normalize_input(self, train_data):
        if self.dummy:
            # 入力はダミー変数化する.
            result = numpy.zeros(self.input_number)
            for i, d in enumerate(train_data):
                result[self.l.index(d) + i * len(self.l) ] += 1
            return result
        else:
            return train_data

def test_q_learning():
    result = []
    result.append({'grid':[[0,2], [0,2]], 'action':3, 'point':4, 'agrid': [[2,0], [0,4]]})
    result.append({'grid':[[4,4], [0,0]], 'action':0, 'point':8, 'agrid': [[0,8], [0,2]]})

    # QLearn obj
    ql_obj =  QLearning(4, 4, dummy=True)

    # before train
    data =[[0,2], [0,2]]
    output_vec= ql_obj.get_q_values(data)
    print output_vec

    # train
    ql_obj.train(result)

    # after train
    data =[[0,2], [0,2]]
    output_vec= ql_obj.get_q_values(data)
    print output_vec



if __name__ == '__main__':
    test_q_learning()

