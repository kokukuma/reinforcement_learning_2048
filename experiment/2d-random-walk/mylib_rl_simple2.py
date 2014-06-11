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
from pybrain.rl.learners.valuebased import ActionValueTable, ActionValueNetwork
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA, NFQ#@UnusedImport
from pybrain.rl.experiments import Experiment
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer
from lib.data_source.random_walk_tool import *
from lib.reinforcement_learning.q_learning import QLearning


def training(agent, args):
    print
    print 'training...'
    print 'total : ', args['total_episodes']
    print 'each  : ', args['episodes']

    #agent.learner._setExplorer(EpsilonGreedyExplorer(epsilon=float(args['learning_epsilon'])))
    for i in range(int(args['total_episodes'])):
        # show_valuse(agent.get_q_values,[0,1])
        # show_valuse(agent.get_q_values,[1,0])
        score, turn = play(agent, 'neural', args)
        # show_valuse(agent.get_q_values,[0,1])
        # show_valuse(agent.get_q_values,[1,0])

        if i % int(args['episodes']) == 0 and not i == 0:
            # print 'hist len :', len(agent.history)
            # for d in agent.history:
            #     print 'sate : ', d['observation'],
            #     print 'action : ',d['action']
            agent.learn()
            agent.reset()
            #print agent.history

        # print
        sys.stdout.write("\r   episodes:%s" % (i))
        sys.stdout.flush()

    print
    return agent

#def q_learning_nfq(result_path, **argvs):
def q_learning_nfq(**args):


    # estimate
    best_score = 0
    best_turn  = 1000
    best_agent = None

    score_list = []
    turn_list = []
    #for i in range(2):
    for i in range(50):

        # # agent 初期化
        # # rand = 1.0
        # # learner = NFQ(maxEpochs=100)
        # # learner._setExplorer(EpsilonGreedyExplorer(rand))
        # controller = ActionValueNetwork(12, 4)
        # learner = NFQ()
        # agent = LearningAgent(controller, learner)
        agent = QLearning(12, 4)

        # training
        agent.greedy_rate   = 0.
        print
        print "==========================="
        print 'before training'
        print_state(agent.get_q_values)
        training(agent, args)
        print 'after training'
        print_state(agent.get_q_values)
        agent.greedy_rate   = 0.7
        #agent.learner._setExplorer(EpsilonGreedyExplorer(0.3))

        score, turn = play(agent, 'neural', args, [2,2])

        score_list.append(score)
        turn_list.append(turn)

        print
        print i, int(numpy.mean(score_list)) , max(score_list) , score, turn

        # if i % args['episodes'] == 0:
        #     try:
        #         agent.learn()
        #     except:
        #         pass
        #     finally:
        #         agent.reset()
        #         # rand = fit_greedy(i)
        #         # agent.learner._setExplorer(EpsilonGreedyExplorer(rand))
        #         # if not i == 0 :
        #         #     import sys
        #         #     sys.exit()
        # print i, int(numpy.mean(score_list)) , max(score_list) , score, turn

        if best_score < score or best_turn > turn:
            best_score = score
            best_turn  = turn
            best_agent = agent
        with open(args['path']+'/result.dump', 'w') as f:
            pickle.dump([score_list, turn_list, best_agent], f)
    print
    print "==========================="
    print 'best score : ', best_score
    print 'best turn : ', best_turn
    print_state(agent)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Incremental Package List Builder')
    parser.add_argument('-p', '--path', default='./')
    parser.add_argument('-t', '--total-episodes', default=100)
    parser.add_argument('-r', '--reward', default=1000)
    parser.add_argument('-i', '--init-state-random', default=False)
    parser.add_argument('-l', '--learning-epsilon', default=1.0)
    parser.add_argument('-e', '--episodes', default=10)
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
