#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class EvaluateError(object):
    def __init__(self):
        pass

    @classmethod
    def split_data(self, vec, proportion=0.5):
        """検証用データの分割"""
        border = len(vec)*proportion
        return vec[:border], vec[border:]

    @classmethod
    def get_rss(self, output_vec, predict_vec):
        """残差平方和"""
        return 0.5 * np.sum((predict_vec - output_vec ) ** 2 )

    @classmethod
    def get_coss_entropy(self, output_vec, predict_vec):
        """交差エントロピー"""
        # TODO: 絶対間違ってるから直す.
        return - np.sum( np.dot(predict_vec.T, np.log(output_vec)) + np.dot((1 - predict_vec).T,  np.log(1 - output_vec)))

