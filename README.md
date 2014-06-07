reinforcement_learning_2048
===========================

## 目標
+ 強化学習の勉強のために, 2048を強化学習で解く.

## 方針
+ 2048自体は, [2048-as-a-service](https://github.com/Semantics3/2048-as-a-service.git )を使って再現.
+ これをローカルで立ち上げて置けば, api叩いて2048を遊ぶことができる.
+ 比較のため, 自作とpybrainを療法で試す.

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


## link
+ [pybrain-rl](docs/pybrain_rl.md)


# 以下, メモ.
---
### 強化学習
+ 強化学習の勉強
  + Temporal Difference learning
  + Q-learning
  + actor-critic
  + 時間軸スムージング学習(Temporal Smoothing learning)



### 良い資料
+ http://sysplan.nams.kyushu-u.ac.jp/gen/edu/RL_intro.html
+ http://www.cs.indiana.edu/~gasser/Salsa/rl.html
+ http://www.cs.indiana.edu/~gasser/Salsa/nn.html

# 実験
+ 内容
  + いろんな手法で, 2049-aiを作成し, その挙動を調査する.

+ 手法による学習の速度, 到達スコアを比較.
  + lookup tableを使ったQ-larning(行動後の履歴を反映: 一般的なQ-learning)
  + lookup tableを使ったQ-larning(過去全ての履歴を反映)
  + NNを使ったQ-learning
  + ロジスティック回帰を使ったQ-learning.
  + DBNを使ったQ-learning


### 考えること.
+ 行動が限定されているような場合は, Q-learningで大丈夫そう.
+ 将来的に得られる報酬の期待値を尤度とした最尤推定. これの位置づけは?



### Q値を再現するNNができてない問題.

+ __連続実数を出力するNNを作るには, 出力関数を何にすればよいのか?__
  + 出力層をy=x, 出力層の更新式を(t-y)x         -> 爆発する
  + sigmoid関数の上限を2->8192にするか？        -> 同上
  + x=0, y=xの漸近線.

+ 学習が進んでも, [0,2],[0,2]に対する行動が決まらないのがおかしい?
  + look-up table だと更新したところ以外は更新されないから.

+ logistic sigmoid関数を使っているのが問題なのか?
  + 放射基底関数を使ってみるか? RBFを使ってみるか?
  + Gauss-Sigmoid を使ってみるか?


+ おそらく, 実数出力のNNが上手くできれば解決する問題.
  + 反復し続けると, 落ちるのはなぜ?
    + パラメータの問題? 中間層数とか, mini-batch数とか.
    + 通常の関数近似ができなくなる時点でNNのやり方が間違っているかも.

  + Q値がマイナスになってるのやっぱおかしくない?
    + 出力層の活性化関数を検討. x=0, y=xの漸近線が望ましいが.
    + sigmoid関数の上限を1->8192にするか？




### 疑問
+ Q-learningの更新アルゴリズム
  + って, 次の状態しか考慮してないけど,
    オンライン学習でなく, 1つの棋譜を学習する状況を考えたら,
    最終的に得られた報酬を今の更新に反映したいけど, どうやる? 意味ある?

+ ニューラルネットワークを使う理由
  + ニューラルネットワークを使う理由は, 連続値を離散値に置き換えるためだろうか?
  + それとも, 次元圧縮のためだろうか?
  + 非線形性を取り込むためだろうか?
  + Salsaのページでは, Q値の更新式は, 通常のQ-learningと同じものを使う.
    つまり, 得られた報酬+に得られる報酬の期待値の最も高い行動を, 正解として学習する.
  + まぁ, 強化学習のポイントが, トレーニングデータがない状態で,
    何を基準に学習するかというところだとすると, これでよいのか.

  + 次元圧縮が主かも.
    + 2048でも, 取りうる状態の数は, 10^16=1e+16くらいある.
    + とても全ての状態を辞書で持っておけるレベルでない.

+ ニューラルネットワークを使ったQ-learningを,
  + 教師データを報酬とQ値によって作成した機械学習ととらえても良いものか?
  + それとも, 多層パーセプトロンの学習方法も変更する必要があるか?
  + 時系列の取り扱いが重要な役割を果たすのであれば,
    Q-tableを固定した状態で, 一気にデータ取って, それを学習するという形はだめかも.


+ オンラインとバッチはどっちが良いだろうか?

+ 入力は, 0-1の範囲に, 正規化する必要があるだろうか.
  + 全てラベルにして, 最大を8192.
  + ダミー変数化すべきか?

+ 出力がQ値なら, 出力がシグモイド関数じゃだめだな...
  + Q値はあくまでもその行動によって将来的に得られる得点の期待値だから,
    ニューラルネットワークの出力も期待値じゃないといけない.
  + ニューラルネットワークで回帰問題を解くためには, 活性化関数を, 何にすれば良いのか?
    + 出力層のみ, y=x
    + ニューロン全部, y=x

+ ニューラルネットワーク自体の学習は, どこまでやるべき?
  + look-up tableの場合は, その結果反映されるから良いとして,
  + ニューラルネットワークの場合は, 反映されるまで反復が必要.
  + しかし遣り過ぎると, 他の
    + 学習すべきデータほど, 反復回数を多くするか?










