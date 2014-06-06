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



#---------------------------------------------------------
# environment / play
#---------------------------------------------------------
def environment(state, action, turn=1):
    """
    action 0:上, 1:下, 2:左, 3:右
    """
    x_number = 5
    y_number = 5

    next_state = state
    reward = 0

    # calc next state
    if action == 0 and not state[1] == 5:
        next_state[1] += 1
    elif action == 1 and not state[1] == 0:
        next_state[1] -= 1
    elif action == 2 and not state[0] == 0:
        next_state[0] -= 1
    elif action == 3 and not state[0] == 5:
        next_state[0] += 1

    # return reward and state
#
    if next_state[0] == 0 and next_state[1] == 0:
        #reward = float(300) / turn
        #reward = 10
        #reward = 100
        reward = 1000

    elif next_state[0] == 5 and next_state[1] == 5:
        reward = 0

    return next_state, reward


# random walk
def play(agent, normalize_type='neural'):
    state = [2, 2]
    # x = numpy.random.randint(4, size=1)[0]
    # y = numpy.random.randint(4, size=1)[0]
    # state = [x+1, y+1]
    # print state
    turn  = 0
    score = 0

    while(1):
        if normalize_type == 'neural':
            observ = normalize(state)
        elif normalize_type == 'table':
            observ = convert(state)

        agent.integrateObservation(observ)
        move = agent.getAction()

        #data = api_move(session_id, move[0])
        state, reward = environment(state, move, turn)

        score += reward
        if turn > 100:
            agent.giveReward(numpy.array([0]))
            return score, turn
        else:
            agent.giveReward(numpy.array([reward]))
            #agent.giveReward(reward)


        turn += 1
        #print turn, state, score, move

        if state in [[0,0],[5,5]]:
            #print data
            break
    return score, turn



#---------------------------------------------------------
# normalize
#---------------------------------------------------------
def convert(state):
    x_number = 6
    y_number = 6
    state_dict = []
    for x in range(x_number):
        for y in range(y_number):
            state_dict.append([x,y])

    return [state_dict.index(state)]

def normalize_old(state):
    """正規化"""
    return [float(state[0])/5, float(state[1])/5]

def normalize(state):
    """ダミー変数化"""
    l = [0, 1, 2, 3, 4, 5]
    result = numpy.zeros(len(state) * len(l))
    for i, d in enumerate(state):
        result[l.index(d) + i * len(l) ] += 1
    return result

def fit_greedy(i):
    """係数変更"""
    if i >= 200:
        return 0.1
    elif i >= 100:
        return 0.2
    elif i >= 50:
        return 0.3
    else:
        return 0.6

def show_valuse(agent,state):
    observ = numpy.array(normalize(state)).ravel()
    print state,
    print ", 0 :", agent.module.getValue(observ, 0),
    print ", 1 :", agent.module.getValue(observ, 1),
    print ", 2 :", agent.module.getValue(observ, 2),
    print ", 3 :", agent.module.getValue(observ, 3)


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

def print_state(agent, normalize_type='neural'):
    x_number = 6
    y_number = 6
    state_dict = []
    print '----------------------------'
    for y in reversed(range(y_number)):
        print y, " : ",
        for x in range(x_number):
            res = []

            if normalize_type == 'neural':
                observ = normalize([x, y])
            elif normalize_type == 'table':
                observ = convert([x,y])

            res.append(agent.module.getValue(observ, 0))
            res.append(agent.module.getValue(observ, 1))
            res.append(agent.module.getValue(observ, 2))
            res.append(agent.module.getValue(observ, 3))

            if res.index(max(res) ) == 0:
                print "↑",
            elif res.index(max(res) ) == 1:
                print "↓",
            elif res.index(max(res) ) == 2:
                print "←",
            elif res.index(max(res) ) == 3:
                print "→",
        print

#---------------------------------------------------------
# learner
#---------------------------------------------------------
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

