#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt

def plot(fig, data):
    ax  = fig.add_subplot(111)
    ax.plot([x for x in range(len(data))], [x for x in data])

def main():
    with open('./score.dump') as f:
        score_list = pickle.load(f)
    fig = plt.figure()
    plot(fig, score_list)
    plt.show()

if __name__ == '__main__':
    main()
