#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

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
from main import api_game_start, api_move, print_state

def play(agent):
    start_data = api_game_start()

    session_id = start_data['session_id']
    grid      = start_data['grid']

    turn   = 0
    result = []
    while(1):
        # agent にはどんな形で渡せばよい?
        #   lib/pybrain/rl/experiments/experiment.py
        #   lib/pybrain/rl/environments/mazes/tasks/mdp.py
        agent.integrateObservation(numpy.array(grid).ravel())
        move = agent.getAction()

        data = api_move(session_id, move[0])

        if grid == data['grid']:
            agent.giveReward(numpy.array([-50]))
            rand_move = numpy.random.randint(4, size=1)[0]
            data = api_move(session_id, rand_move)
        else:
            agent.giveReward(numpy.array([data['points']]))

        turn += 1
        grid = data['grid']

        print_state(turn, data)
        if  data['over']:
            print
            #print data
            break

    return data['score']

def main():

    # 2048の全ての状態を保存するのは無理でしょ.
    #   14^16通りの状態があるよね.
    #controller = ActionValueTable(16, 4)
    #learner = Q()
    #controller.initialize(1.)

    controller = ActionValueNetwork(16, 4)
    learner = NFQ()
    #learner._setExplorer(EpsilonGreedyExplorer(0.0))
    agent = LearningAgent(controller, learner)

    score_list = []
    for i in range(10000):
        # if os.path.exists('./agent.dump'):
        #     with open('./agent.dump') as f:
        #         agent = pickle.load(f)

        print i, 'playing ...'
        score = play(agent)
        score_list.append(score)

        # ここで,
        #   TypeError: only length-1 arrays can be converted to Python scalars
        #   pybrain/rl/learners/valuebased/q.py
        #   => learnerをQからNFQにしたら行けた.
        #   => http://stackoverflow.com/questions/23755927/pybrain-training-a-actionvaluenetwork-doesnt-properly-work
        print i, 'learning ...'
        agent.learn()
        agent.reset()

        print i, 'evaluate sample ...'
        data =[[0,0,0,0], [0,0,0,0], [0,0,0,2], [0,0,0,2]]
        agent.integrateObservation(numpy.array(data).ravel())
        move = agent.getAction()
        print "                           ",i, int(numpy.mean(score_list)) , max(score_list), move

        if i % 20 == 0:
            print i, 'saving ...'
            with open('./agent.dump', 'w') as f:
                pickle.dump(agent, f)
            with open('./score.dump', 'w') as f:
                pickle.dump(score_list, f)

if __name__ == '__main__':
    main()

