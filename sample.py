#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
https://github.com/Semantics3/2048-as-a-service

start server
    cd /Users/karino-t/repos/2048-as-a-service
    node index.js

"""

from collections import defaultdict
import numpy
import datetime
import sys
import re
import json
import urllib2

#HOST="http://2048.semantics3.com"
HOST="http://0.0.0.0:8080"

def predict_next(state):
    return numpy.random.randint(4, size=1)[0]

def api_game_start():
    response = urllib2.urlopen(HOST+'/hi/start/json')
    start_data=json.loads(response.read() )
    session_id = start_data['session_id']
    return  start_data

def api_move(session_id, move):
    response = urllib2.urlopen(HOST+'/hi/state/%s/move/%s/json' % (session_id, move))
    return json.loads(response.read() )

def print_state(turn, data):
    info = '\r turn:%s, score:%s' % (turn, data['score'])
    sys.stdout.write(info)
    sys.stdout.flush()

def main():

    start_data = api_game_start()

    # game start
    print "game start"
    print "session id : ", start_data['session_id']

    # game start
    session_id = start_data['session_id']
    grid      = start_data['grid']

    turn = 0
    while(1):
        # find next move
        move = predict_next(grid)

        # move
        data = api_move(session_id, move)
        grid = data['grid']

        # move
        # print_state(turn, data)
        print move, data['grid']

        if  data['over']:
            print
            print data
            break

        turn += 1

if __name__ == '__main__':
    main()




