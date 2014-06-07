#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from scipy import * #@UnusedWildImport
import pylab
import numpy
import os

#from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners.valuebased import ActionValueTable, ActionValueNetwork
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, SARSA, NFQ#@UnusedImport
from pybrain.rl.experiments import Experiment
from main import api_game_start, api_move, print_state, api_simple_game_start
from pybrain.rl.explorers.discrete.egreedy import EpsilonGreedyExplorer

from lib.random_walk_tool.py import *


def training(agent):
    print 'train'
    agent.learner._setExplorer(EpsilonGreedyExplorer(epsilon=1.0))
    #agent.learner._setExplorer(EpsilonGreedyExplorer(epsilon=1.0, decay=0.5))
    for i in range(100):
        print i
        show_valuse(agent,[0,1])
        show_valuse(agent,[1,0])
        score, turn = play(agent, 'neural')
        show_valuse(agent,[0,1])
        show_valuse(agent,[1,0])

        if i % 10 == 0 and not i == 0:
            agent.learn()
            agent.reset()
    return agent

def q_learning_nfq(result_path):

    controller = ActionValueNetwork(12, 4)

    learner = NFQ()
    #learner = NFQ(maxEpochs=100)
    # rand = 1.0
    # learner._setExplorer(EpsilonGreedyExplorer(rand))
    agent = LearningAgent(controller, learner)

    # training
    print_state(agent)
    training(agent)
    print_state(agent)
    agent.learner._setExplorer(EpsilonGreedyExplorer(0.3))

    # if os.path.exists('./agent.dump'):
    #     with open('./agent.dump') as f:
    #         agent = pickle.load(f)

    # estimate
    score_list = []
    turn_list = []
    for i in range(500):
        score, turn = play(agent, 'neural')

        score_list.append(score)
        turn_list.append(turn)

        if i % 10 == 0:
            try:
                agent.learn()
            except:
                pass
            finally:
                agent.reset()
                # rand = fit_greedy(i)
                # agent.learner._setExplorer(EpsilonGreedyExplorer(rand))

                # if not i == 0 :
                #     import sys
                #     sys.exit()

        print i, int(numpy.mean(score_list)) , max(score_list) , score, turn

        with open(result_path+'/agent.dump', 'w') as f:
            pickle.dump(agent, f)
        with open(result_path+'/score.dump', 'w') as f:
            pickle.dump([score_list, turn_list], f)

    print_state(agent)




def q_learning_table():
    controller = ActionValueTable(36, 4)
    learner = Q()
    controller.initialize(1.)

    agent = LearningAgent(controller, learner)

    score_list = []
    turn_list  = []
    # neural側のトレーニング分 +100
    for i in range(600):
        print_state(agent, 'table')

        score, turn = play(agent, 'table')
        score_list.append(score)
        turn_list.append(turn)

        agent.learn()
        agent.reset()

        print i, int(numpy.mean(score_list)) , max(score_list), score, turn

        with open('./agent.dump', 'w') as f:
            pickle.dump(agent, f)
        with open('./score.dump', 'w') as f:
            pickle.dump([score_list, turn_list], f)



if __name__ == '__main__':
    import sys
    argvs = sys.argv
    if len(argvs) == 0:
        result_path = './'
    else:
        result_path =  argvs[1]
    print result_path

    # NFQ
    q_learning_nfq(result_path)

    # Q-learning
    #q_learning_table()

