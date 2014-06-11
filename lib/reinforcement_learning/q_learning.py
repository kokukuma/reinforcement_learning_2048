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
        self.episode     = []
        self.episodes    = []
        self.train_error = []
        self.valid_error = []

    def agent_observation(self, obs):
        self.last_memory = {'observation': obs, 'action': None, 'reward': None}

    def agent_action(self, action):
        self.last_memory['action']  = action

    def agent_reward(self, reward):
        self.last_memory['reward']  = reward
        self.history.append(self.last_memory)
        self.episode.append(self.last_memory)

        self.last_memory = {}

    def agent_reset(self):
        self.history = []
        self.last_memory = {}

    def agent_save_episode(self):
        self.episodes.append(self.episode)
        self.episode = []

    def print_experience(self):
        """ agentの経験を集計して, summay出すやつ.
            LogAgent classに入れるべきではない気がする.
        """
        print
        print '## print agent experience'
        print 'episode number      : %d ' %  len(self.episodes)

        turn_list  = []
        score_list = []
        for episode in self.episodes:
            turn_list.append(len(episode))
            score_list.append(sum([x['reward'] for x in episode]))

        print 'average episode len : %f(%f)' %  (numpy.average(turn_list),  numpy.std(turn_list))
        print 'average score(std)  : %f(%f)' % (numpy.average(score_list), numpy.std(score_list))
        print
        print '## NN train summay'
        print 'average train error : %f' %(numpy.average([x for x in self.train_error]))
        print 'average valid error : %f' %(numpy.average([x for x in self.train_error]))
        print


class QLearning(LogAgent):
    def __init__(self, input_number, output_number, dummy=False):
        LogAgent.__init__(self, input_number, output_number)

        self.greedy_rate   = 0.3
        self.alpha         = 0.5
        self.ganmma        = 0.9
        self.lastobs       = None
        self.input_number  = input_number
        self.output_number = output_number

        #self.q_mat = numpy.ones((input_number, output_numbe))
        #self.mlnn = MultiLayerNeuralNetwork( [self.input_number, 5, self.output_number],
        # pybrain型network
        self.mlnn = MultiLayerNeuralNetwork( [self.input_number+self.output_number, self.input_number+self.output_number , 1],
                                            threshold=0.5,
                                            start_learning_coef=0.1,
                                            sigmoid_alpha=10,
                                            print_error=False,
                                            mini_batch=50,
                                            epoch_limit=50,
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
            # pybrain型network
            output_vec = self.get_q_values(input_list)
            action = list(output_vec).index(max(output_vec))
            q_vaue = max(output_vec)

        self.agent_action(action)
        if q_value:
            return action, q_vaue
        else:
            return action

    def convert_input(self, input_list, action):
        action_vec = numpy.array([[ 1. if x == action else 0. for x in range(self.output_number)]])
        inp = numpy.append(input_list, action_vec)
        return numpy.array([inp])

    def get_q_values(self, input_list, action=None):
        if action == None:
            q_values = []
            for i in range(self.output_number):
                inp = self.convert_input(input_list, i)
                q_values.append(self.mlnn.predict(numpy.array(inp).ravel()) )
            return q_values
        else:
            inp = self.convert_input(input_list, action)
            return self.mlnn.predict(numpy.array(inp).ravel())

    def reset(self):
        self.agent_reset()
        return

    def learn(self, learn_count=5):
        train_data = self.history

        # ニューラルネットワークのトレーニングデータの形に変換
        input_data , output_data = self.change_format(train_data)
        #input_data , output_data = self.change_format_total_score(train_data)

        # ニューラルネットワークの学習
        for i in range(learn_count):
            error_hist, valid_hist = self.mlnn.train_multi(input_data , output_data)
            self.train_error += [x[1] for x in error_hist]
            self.valid_error += [x[1] for x in valid_hist]
        return error_hist

    def change_format(self, train_data):
        """pybrain型network用"""
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
            before_q_value = self.get_q_values(_observation, _action)

            # 行動後の状態におけるQ値
            after_q_values = []
            for i in range(self.output_number):
                tmp = self.get_q_values(data['observation'], i)
                after_q_values.append(tmp)

            # 特定の行動のQ値を更新する.
            #print self.alpha *(_reward + self.ganmma * max(after_q_values)[0] - before_q_value[0] )
            before_q_value[0] += self.alpha * (_reward + self.ganmma * max(after_q_values)[0]  - before_q_value[0])

            #train_data_input.append(numpy.array(data['grid']).ravel())
            input_data = numpy.array(self.convert_input(_observation, _action)).ravel()

            # for next
            lastexperience = data
            #print input_data, _reward,numpy.array(before_q_value)

            #
            #train_data_input.append(self.normalize_input(input_data))
            train_data_input.append(input_data)
            train_data_output.append(numpy.array(before_q_value))

        return numpy.array(train_data_input) , numpy.array(train_data_output)

    def change_format_old(self, train_data):
        """pybrain型network用"""
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

def test_q_learning():
    # QLearn obj
    ql_obj =  QLearning(5, 2)

    # before train
    data =[[0,0,1,0,0]]
    print ql_obj.get_q_values(data, 0)
    print ql_obj.get_q_values(data, 1)

    # train
    for i in range(20):
        grid = [0,0,1,0,0]
        while(1):
            ql_obj.integrateObservation([grid])
            action = ql_obj.getAction()

            # move 1
            if action == 0:
                grid.append(0)
                grid.pop(0)
            else:
                grid.insert(0, 0)
                grid.pop(5)

            # get reward
            if grid.index(1) == 0:
                ql_obj.giveReward(100)
            else:
                ql_obj.giveReward(0)
            if grid.index(1) in (0,4):
                break
        ql_obj.agent_save_episode()

    ql_obj.learn()
    ql_obj.print_experience()

    # after train
    data =[[0,0,1,0,0]]
    print ql_obj.get_q_values(data, 0)
    print ql_obj.get_q_values(data, 1)

    #print ql_obj.mlnn.weights

if __name__ == '__main__':
    test_q_learning()

