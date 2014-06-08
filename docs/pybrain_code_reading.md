pybrain_code_reading.md
===========================

## reinforcement learning
+ entrypoint
  + docs/tutorials/rl.py
  + NNを使ったQ-learningではないが, 以下のように読み替えればOK.
    + ActionValueTable -> ActionValueNetwork
    + Q -> NFQ

+ [agent] lib/pybrain/rl/agents/learning.py
  + 環境のやり取りをするagent(状態の入力, アクションの取得, 学習を行う.)
  + instanceを作るときに, module/learnerを設定.
  + moduleは学習した結果を保存する器. ActionValueTable/ActionValueNetwork を設定.
  + learnerは学習ロジック.
  + LoggingAgentを継承していて, 観測した状態や受け取った報酬を保存している.
  + learn
    + 指定した回数だけlearnを実行. それまでに記録した情報で, 学習を繰り返す.
    + EpisodicLearner/learnEpisodes でlearnを実行
    + これは,nfq classで実装されているものを呼んでいるのだろう.
  + reset
    + log/module/learnerどちらもresetを実行.
    + log     : 登録した状態, 行動, 報酬が削除される.
    + module  : 今までにmoduleの中に溜め込んだbufferを削除.
    + learner : なにもresetしてない. (深いところで継承されてるEpisodicLearner)
  + _end_

+ [agent] lib/pybrain/rl/agents/logging.py
  + 観測した状態, 取った行動, 受け取った報酬を保存している.
  + integrateObservation / getAction / giveReward
  + integrateObservation / getActionではそのときの状態を一時保存. (getActionでの保存は, なぜか継承先で保存.)
  + giveReward でaddSampleで登録. きっといい感じのデータ構造になっているんだろう.
  + _end_

+ [module] lib/pybrain/rl/learners/valuebased/interface.py
  + ActionValueTable/ActionValueNetwork
  + ActionValueNetwork : 状態の数と行動の数を指定して作成
  + 多層パーセプトロン構造
    + 入力:状態+行動数
    + 中間:状態+行動数 (SigmoidLayer)
    + 出力:1           (LinearLayer)
    + 状態と行動を入力すると, Q値を返す想定.
  + inputやoutputを内部のbufferに溜め込んでいる...
    + 何のためかいまいち分からん. 読みづらくなってるだけとしか思えん.
    + backprop.pyのcalcDerivsとか.
  + _end_


+ [learner] lib/pybrain/rl/learners/valuebased/nfq.py
  + NFQ algorithm(Neuro-fitted Q-learning)
  + 最大反復回数 : 20回, そんなに少なくていいのか...
  + 割引率       : デフォ0.9
  + 学習係数α   : 0.5固定
  + 状態と行動を入力, Q値の更新式で得た値をトレーニングデータとして,
    多層パーセプトロンの学習を行う.
  + 中で, ValueBasedLearnerを継承. EpsilonGreedyExplorerをセットする.
  + 学習方法は, 通常のBackpropTrainerではなく, RPropMinusTrainerが使われている.
  + _end_


+ [learner] lib/pybrain/supervised/trainers/backprop.py
  + 学習係数learningrateのデフォ, 0.01とかすげー小さい.
  + lrdecay=1.0, 学習係数を減衰させるか. デフォさせない.
  + momentumもついてる.
  + weightdecay rateってなんだ?
  + 与えられたデータを, trainingData/validationDataに分けている.
    + 反復の度にエラーを検査して, 最も良い重み, エラー, epochを保存している.
    + 最後の反復が終わったら, もっともいい結果を選択して返す.
    + trainerror / validateerrorの２つがある.

  + backpropのやり方
    + buildnetwork当たりが結構深い..
    + calcDerivsでderivsを求め, derivsから重み(params)を更新している.
      + input/output/trainから, backpropされたときのエラーをmodule内部に保存.
    + derivsの計算
      + self.module.backActivate
        + network/_backwardImplementation
          + lib/pybrain/structure/networks/feedforward.py
    + 各層の違い
      + 各層のforward/backword方法を定義しておき, 接続順と逆に実行していく.
        + network/_backwardImplementation
        + lib/pybrain/structure/networks/feedforward.py
  + 残る疑問点
    + importanceってなんだ?
    + module.derivsって結局, 重みの更新量?
  + _end_


+ [learner] lib/pybrain/supervised/trainers/rprop.py
  + backpropを継承して作られる.
  + 1epoch中のトレーニングを行う関数(train)だけoverride.
  + RPropMinusTrainerは何が違うのか?
    + 結局backpropのbatch=Trueと同じロジックに見えるんだが.
  + Rpropにも種類あり(Rprop+, Rprop-, iRprop-), ここでは, Rprop-が使われてる.
    + http://d.hatena.ne.jp/keyword/RPROP
  + backpropの派生
    + Quickprop, RPropなどがあるっぽいが...
    + varient of the Sparse Distributed Memory の方が良いって人も.
  + _end_

+ [explorer] lib/pybrain/rl/explorers/discrete/egreedy.py
  + 学習した内容をもとに, 行動を選択する.
  + ただし, epsilon(0.3)の割合で, ランダムに行動する.
  + また, 選択のたびに, decay(0.9999)の割合で, epsilonを小さくしている.
  + つまり, だんだん貪欲になっていくという形.
  + _end_


