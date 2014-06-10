reinforcement_learning_2048
===========================

## 目標
+ 強化学習の勉強のために, 2048を強化学習で解く.

## 方針
+ 2048自体は, [2048-as-a-service](https://github.com/Semantics3/2048-as-a-service.git )を使って再現.
+ これをローカルで立ち上げて置けば, api叩いて2048を遊ぶことができる.
+ 比較のため, 自作とpybrainを試す.
+ 対象モデルは, 2次元ランダムウォーク, 2048-3x3, 2048-4x4

## 起動方法
+ 2048_as_a_service
  + node index.js 
+ 2048_ai
  + python main.py
  + python pybrain_rl.py

## 計算環境の構築
+ python
  + sudo apt-get install python2.7-dev
+ numpy/scipy/matplotlib
  + sudo apt-get install python-setuptools
  + sudo apt-get install python-numpy
  + sudo apt-get install python-scipy
  + sudo apt-get install python-matplotlib
+ pybrain
  + http://pybrain.org/docs/quickstart/installation.html
+ 2048
  + https://github.com/Semantics3/2048-as-a-service
  + https://github.jp.klab.com/karino-t/reinforcement_learning_2048

## pybrainに関する調査
+ [pybrain-rl](docs/pybrain_rl.md)
  + pybrainで強化学習やるまでに悪銭苦闘したこと.
+ [pybrain-code-reading](docs/pybrain_code_reading.md)
  + pybrainで使われてる強化学習の手法を読んだときのメモ.


## 2次元ランダムウォーク
+ Q-learningの検証のため, 簡単なモデルで試す.
+ pybrainで実行してみた結果
  + [2d-random-walk](docs/2d_random_walk.md)
  + 実行
    + ```python experiment/2d-random-walk/pybrain_rl_simple2.py```
+ mylibで実行してみた結果


