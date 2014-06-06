#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle
import matplotlib.pyplot as plt

def plot(fig, data):
    ax  = fig.add_subplot(111)
    ax.plot([x for x in range(len(data))], [x for x in data])

def main():
    argvs = sys.argv
    with open(argvs[1]) as f:
        tmp = pickle.load(f)
        score_list = tmp[0]
        turn_list  = tmp[1]
        print score_list

    fig = plt.figure()
    plot(fig, score_list)
    plt.show()

    fig2 = plt.figure()
    plot(fig2, turn_list)
    plt.show()

if __name__ == '__main__':
    main()
