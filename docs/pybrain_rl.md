pybrain_rl.md
===========================

## pybrainでちゃんと強化学習できるようになるまでに試したことのめも
+ 見られる現象
  + 途中から報酬がない方で固定されてしまう.

+ もっと簡単な問題を解いてみる.
  + 2048 - 3x3 で実施.
    + => 6000ゲームやってみたが, 結果はランダムと変わらず.
  + 2次元random walk
    + => Q-learningは上手く言ったぽいが, NFQがだめ.

+ 問題は違うがこの辺では, 1000万回で学習が収束したという.
  + 10,000回じゃ足りない?
  + http://yabsv.jks.ynu.ac.jp/PaperPDF/ROBOMEC2011/sakai2010.pdf
  + => 3x3を10万回まわしてみたが微妙.

+ 入力値の範囲は?
  + 0-1, -1-1に正規化する必要があるか?
  + 中間層がsigmoid関数なら, 例えば入力値4,5の違いは出ないのではないか?
  + 状態や報酬ではなく, Neural Networkの入力が0-1に正規化されるように調整する.
  + => random walk , lookuptableを使ったQ-learningで, マイナス入れない方が上手くいった.
  + => NFQの方はそんなに変わらない.

+ コストか報酬か?
  + 報酬で計算してる? それともコストで計算してる?
  + => Qの更新で, maxを取っているところを見ると, 報酬で計算しているように思われる.

+ 学習に関する係数の変更
  + episodes変更 :  1 -> 100
  + epoch変更    : 20 -> 100
  + => 遅くなるだけであんまり意味ないような感じ.

+ スタート地点をランダムにしたらどうなる?
  + => 遅くなるだけであんまり意味ないような感じ.

+ データが干渉しているために起こるのであれば.
  + トレーニングデータを1gameじゃなくて, もっとためてから学習しては?
  + Gauss-Sigmoid ニューラルネットワークにしては?
  + => トレーニングデータを10game分ためて学習すると上手くいった...1回だけ...

+ 得点の与え方
  + ゴールまでのステップ数で得点変えるよりも, 変えない方が上手くいった.

+ 最初に典型的な進み方を学習させて置いたらどうなる?

+ 強化学習のレベルでなく, ニューラルネットワークのレベルで考えると良いかも.
  + 最初に更新される点は, 報酬が得られる増すの周辺のみ.
  + その２つが意図通りの更新となっているかどうかを確認する.

+ 報酬のレンジ
  + 初期のQ値を超えるくらい.

+ トレーニングと実行の分離
  + これやって強化学習といえるのか
  + これのときは, 完全ランダムでやった方がいいか?
  + 状態が多いときは, 完全ランダムじゃたどり着けない場所がありそうだから,
    焼き鈍し法見たいにした方がいいかも.
  + 最高得点までは, 0.3, それ以降は1.0で探索するか?
    でも, それは局所解の最高得点かもしれない.

+ 疑うべきポイント
  + まず, NNの入力値-出力値がモデルと一致しているものだろうか?
  + 得られたトレーニングデータは十分なものだろうか(中に正解が含まれているか?)
  + NNが収束しているか?

+ 現状一番上手く行く組み合わせ
  + game 10
  + 報酬は最後に一気に.
  + 入力値は, ダミー変数化
  + 割引率はデフォ(0.99999).
  + trainを入れて, そのときは,  greedly =1.0にする.

