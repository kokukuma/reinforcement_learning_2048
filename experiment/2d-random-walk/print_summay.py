#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
+ 再起的にresult.dumpを探して, summaryを表示する.
+ markdown の表形式で表示する.
+ 最高得点のエージェントの経験を出力
result + base + network
       |      |
       |      + table
       +
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy
import sys
import pickle
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from lib.data_source.random_walk_tool import *

def plot(fig, data):
    ax  = fig.add_subplot(111)
    ax.plot([x for x in range(len(data))], [x for x in data])

def get_results(dump_data):
    result = {}
    score_list = dump_data[0]
    turn_list  = dump_data[1]
    agent      = dump_data[2]

    result['score_ave']   =  numpy.average(score_list[-50:])
    result['score_std']   =  numpy.std(score_list[-50:])
    result['turn_ave']    =  numpy.average(turn_list[-50:])
    result['turn_std']    =  numpy.std(turn_list[-50:])
    result['score_list']  =  score_list
    result['turn_list']   =  turn_list
    result['agent']       =  agent

    return result

def main():
    argvs = sys.argv
    ignore_path = []
    #ignore_path = ['result/base/table', 'result/input_format/normalized']

    # search dump files
    dump_files = defaultdict(dict)
    for root, dirs, files in os.walk(argvs[1]):
        if not root in ignore_path:
            for path in files:
                if path in ['result.dump']:
                    dump_files[root]['score'] = root + '/' + path

    # get result
    results = defaultdict(dict)
    for key, data in dump_files.items():
        with open(data['score']) as f:
            tmp = pickle.load(f)

        results[key] = get_results(tmp)

    print
    print '## best Q-value action'
    for key, data in sorted(results.items()):
        #print "%10s, %10s(%s), %10s(%s) " % (key , data['score_ave'], data['score_std'], data['turn_ave'], data['turn_std'])
        print
        print '+ ', key
        print_state(data['agent'].get_q_values, normalize_type='neural')
    print

    # print result
    print '## summary score/step '
    print
    print_summary_table(results)

    print '## summary agent training state'
    print
    print_agent_data(results)

    # save plot
    for key, data in results.items():
        fig = plt.figure()
        plot(fig, data['score_list'])
        plt.savefig(key + '/score_list.png')
        fig2 = plt.figure()
        plot(fig2, data['turn_list'])
        plt.savefig(key + '/turn_list.png')



def print_summary_table(results):
    print '|%20s|%20s|%20s|' % ('path', 'average score', 'average step')
    print '|%20s|%20s|%20s|' % ('-' * 20, '-' * 20, '-' * 20)

    for key, data in results.items():
        print '|%20s| %8.2f(%8.2f) | %8.2f(%8.2f) |' % (key, data['score_ave'], data['score_std'], data['turn_ave'],data['turn_std'])
    print

def print_agent_data(results):
    print '|%20s|%20s|%20s|%10s|%10s|' % ('path', 'train data ave-score', 'train data ave-step', 'train err', 'valid err')
    print '|%20s|%20s|%20s|%10s|%10s|' % ('-' * 20, '-' * 20, '-' * 20, '-' * 10, '-' * 10)

    for key, data in results.items():
        agent = data['agent']
        turn_list  = []
        score_list = []
        for episode in agent.episodes:
            turn_list.append(len(episode))
            score_list.append(sum([x['reward'] for x in episode]))

        train_score_ave = numpy.average(score_list)
        train_score_std = numpy.std(score_list)
        train_step_ave  = numpy.average(turn_list)
        train_step_std  = numpy.std(turn_list)
        nn_train_error  = numpy.average([x for x in agent.train_error])
        nn_valid_error  = numpy.average([x for x in agent.valid_error])

        print '|%20s| %8.2f(%8.2f) | %8.2f(%8.2f) | %8.2f | %8.2f |' % (key, train_score_ave,
                                                                             train_score_std,
                                                                             train_step_ave,
                                                                             train_step_std,
                                                                             nn_train_error,
                                                                             nn_valid_error)
    print


if __name__ == '__main__':
    main()
