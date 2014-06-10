#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
https://github.com/Semantics3/2048-as-a-service

start server
    cd /Users/karino-t/repos/2048-as-a-service
    node index.js

"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from lib.q_learning import QLearning
import sys
import re
import json
import urllib2
import numpy

#HOST="http://2048.semantics3.com"
HOST="http://0.0.0.0:8080"

def api_game_start():
    response = urllib2.urlopen(HOST+'/hi/start/json')
    start_data=json.loads(response.read() )
    session_id = start_data['session_id']
    return  start_data

def api_simple_game_start():
    #print HOST+'/hi/start/size/2/tiles/2/victory/4/rand/4/json'
    response = urllib2.urlopen(HOST+'/hi/start/size/3/tiles/2/victory/13/rand/2/json')
    start_data=json.loads(response.read() )
    session_id = start_data['session_id']
    return  start_data

def api_move(session_id, move):
    response = urllib2.urlopen(HOST+'/hi/state/%s/move/%s/json' % (session_id, move))
    return json.loads(response.read() )

def print_state(turn, data):
    info = '\r                           turn:%s, score:%s' % (turn, data['score'])
    sys.stdout.write(info)
    sys.stdout.flush()


def play(ql_obj):

    start_data = api_game_start()

    # # game start
    # print "game start"
    # print "session id : ", start_data['session_id']

    # game start
    session_id = start_data['session_id']
    grid      = start_data['grid']

    turn   = 0
    result = []
    while(1):

        # find next move
        move, q_value = ql_obj.predict_next(grid, greedy=True)


        # move
        data = api_move(session_id, move)
        result.append({'grid':grid, 'action':move, 'point':data['points'], 'agrid': data['grid']})

        grid  = data['grid']


        # 状況表示と終了判定
        print_state(turn, data)
        #print move, data['grid']
        #if  data['over'] or turn > 10:
        if  data['over']:
            print
            #print data
            break

        turn += 1

    return data['score'] , result


def main():

    # ダミー変数化のため, [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # 入力素子数は, 16×14にされる.
    ql_obj =  QLearning(16, 4, dummy=False)

    max_score = 0
    score_list = []
    for i in range(10000):

        score, result = play(ql_obj)

        # Q-learning
        ql_obj.train(result)

        score_list.append(score)

        # print weight
        data =[[0,0,0,0], [0,0,0,0], [0,0,0,2], [0,0,0,2]]
        output_vec= ql_obj.get_q_values(data)
        print i, numpy.mean(score_list) , max(score_list), output_vec

if __name__ == '__main__':
    main()




