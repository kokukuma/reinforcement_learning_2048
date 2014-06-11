#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import numpy
from lib.multilayer_neural_network.multilayer_perceptron import  MultiLayerNeuralNetwork
from lib.multilayer_neural_network.layer.linear_layer import LinearLayer
from lib.multilayer_neural_network.layer.sigmoid_layer import SigmoidLayer
from lib.multilayer_neural_network.layer.softmax_layer import SoftmaxLayer

class LogAgent(object):
    def __init__(self, input_num, output_num):
        self.history = []
        self.last_memory = {}

    def agent_observation(self, obs):
        self.last_memory = {'observation': obs, 'action': None, 'reward': None}

    def agent_action(self, action):
        self.last_memory['action']  = action

    def agent_reward(self, reward):
        self.last_memory['reward']  = reward
        self.history.append(self.last_memory)
        self.last_memory = {}

    def agent_reset(self):
        self.history = []
        self.last_memory = {}


class QLearning(LogAgent):
    def __init__(self, input_number, output_number, dummy=False):
        LogAgent.__init__(self, input_number, output_number)

        self.l = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        self.dummy = dummy
        self.greedy_rate   = 0.3
        self.alpha         = 0.5
        self.ganmma        = 0.9
        self.memory        = {}
        self.lastobs       = None
        if self.dummy:
            self.input_number  = input_number * len(self.l)
        else:
            self.input_number  = input_number
        self.output_number = output_number
        #self.q_mat = numpy.ones((input_number, output_numbe))
        self.mlnn = MultiLayerNeuralNetwork( [self.input_number,  5, self.output_number],
                                            threshold=0.5,
                                            start_learning_coef=0.1,
                                            sigmoid_alpha=10,
                                            print_error=True,
                                            mini_batch=10,
                                            epoch_limit=500,
                                            layer_type=[LinearLayer, SigmoidLayer, LinearLayer],
                                            rprop=True)

    def integrateObservation(self, state_vec):
        self.agent_observation(state_vec)
        self.lastobs = state_vec
        return

    def giveReward(self, reward):
        self.agent_reward(reward)
        return

    def getAction(self, input_list=None, greedy=True, q_value=False):
        if input_list == None:
            input_list = self.lastobs

        if greedy and self.greedy_rate < numpy.random.random():
            action = numpy.random.randint(self.output_number, size=1)[0]
            q_vaue = 0
        else:
            output_vec = self.get_q_values(input_list)
            action = list(output_vec).index(max(output_vec))
            q_vaue = max(output_vec)

        self.agent_action(action)
        if q_value:
            return action, q_vaue
        else:
            return action

    def get_q_values(self, input_list, action=None):
        input_mat = numpy.array(input_list)
        #output_vec = self.mlnn.predict( input_mat.ravel() )
        #output_vec = self.mlnn.predict( self.normalize_input(input_mat.ravel()))
        output_vec = self.mlnn.predict(input_mat.ravel())

        if action == None:
            return map((lambda x: x if x >0 else 0), output_vec)
        else:
            return output_vec[action] if output_vec[action]>0 else 0

    def reset(self):
        self.agent_reset()
        return

    def learn(self):
        train_data = self.history
        # print '========================================================='
        # for i in train_data:
        #     print i['observation'], i['action'] ,i['reward']
        # print '========================================================='

        # ニューラルネットワークのトレーニングデータの形に変換
        input_data , output_data = self.change_format(train_data)
        #input_data , output_data = self.change_format_total_score(train_data)

        # ニューラルネットワークの学習
        error_hist = self.mlnn.train_multi(input_data , output_data)

        return error_hist

    def change_format(self, train_data):
        train_data_input  = []
        train_data_output = []
        lastexperience = None
        for data in train_data:
            if not lastexperience:
                lastexperience = data
                continue
            _observation = lastexperience['observation']
            _action      = lastexperience['action']
            _reward      = lastexperience['reward']

            # 行動前の状態におけるQ値
            q_vec = self.get_q_values(_observation)

            # 行動後の状態におけるQ値
            move, q_value = self.getAction(data['observation'], q_value=True)

            # 特定の行動のQ値を更新する.
            q_vec[_action]  += self.alpha * (_reward + self.ganmma * q_value  - q_vec[_action])

            #train_data_input.append(numpy.array(data['grid']).ravel())
            input_data = numpy.array(_observation).ravel()

            # for next
            lastexperience = data

            #
            #train_data_input.append(self.normalize_input(input_data))
            train_data_input.append(input_data)
            train_data_output.append(numpy.array(q_vec))

        return numpy.array(train_data_input) , numpy.array(train_data_output)

    # def change_format_total_score(self, train_data):
    #     train_data_input  = []
    #     train_data_output = []
    #     q_list = []
    #     for data in reversed(train_data):
    #         # 行動前の状態におけるQ値
    #         q_vec = self.get_q_values(data['grid'])
    #
    #         # 行動後の状態におけるQ値
    #         move, q_value = self.getAction(data['agrid'])
    #
    #         # 特定の行動のQ値を更新する.
    #         q_list = [float(q) * self.ganmma for q in q_list]
    #         q_list.append(data['point'])
    #         q_vec[data['action']]  += self.alpha * (sum(q_list) + self.ganmma * q_value  - q_vec[data['action']])
    #         #print sum(q_list)
    #
    #         #train_data_input.append(numpy.array(data['grid']).ravel())
    #         input_data = numpy.array(data['grid']).ravel()
    #
    #         train_data_input.append(self.normalize_input(input_data))
    #         train_data_output.append(numpy.array(q_vec))
    #
    #     return numpy.array(train_data_input) , numpy.array(train_data_output)

    # def normalize_input(self, train_data):
    #     if self.dummy:
    #         # 入力はダミー変数化する.
    #         result = numpy.zeros(self.input_number)
    #         for i, d in enumerate(train_data):
    #             result[self.l.index(d) + i * len(self.l) ] += 1
    #         return result
    #     else:
    #         return train_data

def test_q_learning():
    result = []
    result.append({'grid':[[0,0,1,0,0]], 'action':0, 'point':0})
    result.append({'grid':[[0,0,1,0,0]], 'action':1, 'point':0})
    result.append({'grid':[[0,1,0,0,0]], 'action':0, 'point':1000})
    result.append({'grid':[[0,1,0,0,0]], 'action':1, 'point':0})
    result.append({'grid':[[0,0,0,1,0]], 'action':0, 'point':0})
    result.append({'grid':[[0,0,0,1,0]], 'action':1, 'point':0})

    # QLearn obj
    ql_obj =  QLearning(5, 2)

    # before train
    data =[[0,0,1,0,0]]
    output_vec= ql_obj.get_q_values(data)
    print output_vec

    # train
    for data in result:
        ql_obj.integrateObservation(data['grid'])
        action = ql_obj.getAction()
        ql_obj.giveReward(data['point'])
    ql_obj.learn()

    # after train
    data =[[0,0,1,0,0]]
    output_vec= ql_obj.get_q_values(data)
    print output_vec



if __name__ == '__main__':
    test_q_learning()

