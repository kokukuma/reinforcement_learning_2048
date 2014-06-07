#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy import * #@UnusedWildImport
import pylab
import numpy


#---------------------------------------------------------
# environment / play
#---------------------------------------------------------
def environment(state, action, turn=1, args=None):
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
        #reward = 1000
        reward = args['reward']

    elif next_state[0] == 5 and next_state[1] == 5:
        reward = 0

    return next_state, reward


# random walk
def play(agent, normalize_type='neural', args=None):
    if args['init_state_random']:
        x = numpy.random.randint(4, size=1)[0]
        y = numpy.random.randint(4, size=1)[0]
        state = [x+1, y+1]
    else:
        state = [2, 2]
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
        state, reward = environment(state, move, turn, args)

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


def print_state(agent, normalize_type='neural'):
    x_number = 6
    y_number = 6
    state_dict = []
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
