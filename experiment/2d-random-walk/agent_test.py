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

def learning(best_agent):

    # print_experience
    best_agent.print_experience()
    print_state(best_agent.get_q_values)

    agent = QLearning(12, 4)

    # 学習前
    agent.episodes = best_agent.episodes

    # 学習
    for i, episode in enumerate(best_agent.episodes):
        agent.history += episode
        if i % 10 == 0 and not i == 0:
            agent.learn()
            agent.reset()

    # 学習前
    print_state(agent.get_q_values)

    # print_experience
    agent.print_experience()


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Incremental Package List Builder')
    parser.add_argument('-a', '--agent-path', default='./result.dump')
    args = parser.parse_args()


    print args.agent_path
    with open(args.agent_path, 'r') as f:
        tmp = pickle.load(f)
    best_agent = tmp[2]

    learning(best_agent)

