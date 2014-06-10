#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
result + base + network
       |      |
       |      + table
       +
"""

import numpy
import sys
import pickle
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from lib.random_walk_tool import *

def plot(fig, data):
    ax  = fig.add_subplot(111)
    ax.plot([x for x in range(len(data))], [x for x in data])


def main():
    argvs = sys.argv
    ignore_path = ['result/base/table', 'result/input_format/normalized']

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
            score_list = tmp[0]
            turn_list  = tmp[1]
            agent      = tmp[2]

        results[key]['score_ave'] =  numpy.average(score_list[-50:])
        results[key]['score_std'] =  numpy.std(score_list[-50:])
        results[key]['turn_ave']  =  numpy.average(turn_list[-50:])
        results[key]['turn_std']  =  numpy.std(turn_list[-50:])
        results[key]['agent']         =  agent

        # save plot
        fig = plt.figure()
        plot(fig, score_list)
        plt.savefig(key + '/score_list.png')
        fig2 = plt.figure()
        plot(fig2, turn_list)
        plt.savefig(key + '/turn_list.png')


    # print result
    for key, data in sorted(results.items()):
        print "%10s, %10s(%s), %10s(%s) " % (key , data['score_ave'], data['score_std'], data['turn_ave'], data['turn_std'])
        print_state(data['agent'], normalize_type='neural')

if __name__ == '__main__':
    main()
