#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import pickle
from scipy import * #@UnusedWildImport
import pylab
import numpy
from pprint import pprint
from lib.reinforcement_learning.q_learning import QLearning

from lib.data_source.api_2048 import api_game_start, api_move, print_state, api_simple_game_start

def normalize(state):
    """ダミー変数化"""
    l = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    result = numpy.zeros(len(state) * len(l))
    for i, d in enumerate(state):
        result[l.index(d) + i * len(l) ] += 1
    return result

def play(agent):
    #start_data = api_game_start()
    start_data = api_simple_game_start()

    session_id = start_data['session_id']
    grid      = start_data['grid']

    turn   = 0
    result = []
    while(1):
        #print numpy.array(grid).ravel()
        agent.integrateObservation(normalize(numpy.array(grid).ravel()))
        move = agent.getAction()
        data = api_move(session_id, move)

        if grid == data['grid']:
            #agent.giveReward(numpy.array([-50]))
            rand_move = numpy.random.randint(4, size=1)[0]
            data = api_move(session_id, rand_move)
            agent.agent_action(rand_move)

        # その都度報酬を付与するか?
        # 最後に一気にふよするか?
        # それとも両方やるか?
        agent.giveReward(numpy.array([data['points']]))

        turn += 1
        grid = data['grid']

        if  data['over']:
            break

    return data['score'], turn



def training(agent, args):
    print
    print 'training...'
    print 'total : ', args['total_episodes']
    print 'each  : ', args['episodes']

    for i in range(int(args['total_episodes'])):
        score, turn = play(agent)

        agent.agent_save_episode()

        if i % int(args['episodes']) == 0 and not i == 0:
            agent.learn()
            agent.reset()

        # print
        # sys.stdout.write("\r   episodes:%s, turn:%d, score:%d" % (i, turn, score ))
        # sys.stdout.flush()

    #print
    agent.print_experience()
    return agent

def q_learning_nfq(**args):

    # estimate
    best_score = 0
    best_turn  = 1000
    best_agent = None

    score_list = []
    turn_list = []
    for i in range(1):
    #for i in range(50):

        #agent = QLearning(12, 4)
        agent = QLearning(117, 4)

        # training
        agent.greedy_rate   = 0.1
        for i in range(10):
            print
            print "==========================="
            agent.greedy_rate += 0.05 if agent.greedy_rate < 0.7 else 0.7
            training(agent, args)
        agent.greedy_rate   = 0.7

        #score, turn = play(agent, 'neural', args, [2,2])
        score, turn = play(agent)

        score_list.append(score)
        turn_list.append(turn)

        print
        print 'test one play'
        print i, int(numpy.mean(score_list)) , max(score_list) , score, turn

        if best_agent==None or numpy.average(best_agent.train_error) > numpy.average(agent.train_error):
            print 'best train error !'
            best_score = score
            best_turn  = turn
            best_agent = agent
        # if best_score < score or best_turn > turn:
        #         print 'best train error !'
        #         best_score = score
        #         best_turn  = turn
        #         best_agent = agent

        with open(args['path']+'/result.dump', 'w') as f:
            pickle.dump([score_list, turn_list, best_agent], f)
    print
    print "==========================="
    print 'best score : ', best_score
    print 'best turn : ', best_turn

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Incremental Package List Builder')
    parser.add_argument('-p', '--path', default='./')
    parser.add_argument('-t', '--total-episodes', default=1000)
    parser.add_argument('-r', '--reward', default=1000)
    parser.add_argument('-i', '--init-state-random', default=False)
    parser.add_argument('-l', '--learning-epsilon', default=1.0)
    parser.add_argument('-e', '--episodes', default=100)
    parser.add_argument('-d', '--decay', default=0.9)
    args = parser.parse_args()

    # NFQ
    #q_learning_nfq(result_path, **argvs)
    print
    print '============================================'
    print 'NFQ'
    print '============================================'
    pprint(vars(args))
    q_learning_nfq(**vars(args))

    # Q-learning
    #q_learning_table()

