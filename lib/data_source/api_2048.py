#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import urllib2
import json
import sys

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

