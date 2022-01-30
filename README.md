# g_research_crypto

## 概要
毎日400億ドル以上の暗号通貨が取引されています。暗号通貨は投機や投資のための最も人気のある資産の一つですが、乱高下が激しいことが分かっています。価格変動が激しいため、一部の幸運な人は億万長者になり、他の人は大損害を被っている。このような値動きを事前に予測することは可能だったのでしょうか？

このコンペティションでは、機械学習の専門知識を駆使して、人気のある14種類の暗号通貨の短期的なリターンを予測していただきます。私たちは、2018年までさかのぼる数百万行の高頻度市場データを蓄積しており、あなたのモデルを構築するために使用することができます。提出期限が過ぎると、あなたの最終的なスコアは、収集されたライブの暗号データを使用して、次の3ヶ月間で計算されます。

何千人ものトレーダーが同時に活動しているため、ほとんどのシグナルは一過性のものであり、持続的なアルファを見つけることは非常に難しく、オーバーフィッティングの危険性はかなり高くなります。さらに、2018年以降、暗号市場への関心が爆発的に高まっているため、我々のデータにおけるボラティリティと相関構造は高度に非定常である可能性が高いです。合格者は、これらの点に細心の注意を払い、その過程で金融予測の芸術と科学に関する貴重な洞察を得ることができます。

G-Researchは、ヨーロッパを代表する定量的金融調査会社です。私たちは、機械学習、ビッグデータ、そして最先端のテクノロジーを駆使して、市場予測の可能性を長年にわたって追求してきました。ワークフォース向けのデータサイエンスとAI教育に特化したCambridge Sparkは、G-Researchと提携してこのコンペティションを開催しています。コンペティションの紹介は以下からご覧ください。

## データ
このデータセットには、BitcoinやEthereumなど、いくつかの暗号資産の過去の取引情報が含まれています。あなたの挑戦は、それらの将来のリターンを予測することです。

暗号通貨の歴史的な価格は機密ではないため、時系列APIを使用した予測コンペティションとなります。さらに、パブリックリーダーボードターゲットは一般に公開され、コンペティションデータセットの一部として提供されます。多くの人が楽しみながら完璧な投稿をすることを期待しています。したがって、このコンペティションのパブリックリーダーボードには意味がなく、自分のコードをテストしたい人のために便宜的に提供されているに過ぎません。最終的なプライベートリーダーボードは、応募期間終了後に収集された実際の市場データを使用して決定されます。

### ファイル
train.csv - トレーニングセット

    timestamp - その行がカバーする分のタイムスタンプ。
    Asset_ID - 暗号アセットのIDコード。
    Count - この分間に行われた取引の数。
    オープン - その分単位の開始時点の米ドル価格。
    高値 - その分、最も高い米ドル価格。
    低 - その分、最低の米ドル価格です。
    クローズ - 分の終わりの USD 価格.
    取引量 - その分単位で取引された暗号化資産ユニット数。
    VWAP - 1分間のボリューム加重平均価格です。
    ターゲット - 15分の残差リターン。ターゲットの計算方法の詳細については、このノートの「予測と評価」セクションを参照してください。

example_test.csv - 時系列APIによって配信されるデータの例です。

example_sample_submission.csv - 時系列APIから配信されるデータの一例。データはtrain.csvからコピーしたものである。

asset_details.csv - 各Asset_IDのcryptoassetの実名と、各cryptoassetがメトリックで受け取るウェイトを提供する。

gresearch_crypto - 時系列APIファイルのオフライン作業用の非最適化バージョンです。エラーなしで実行するには、Python 3.7とLinux環境が必要な場合があります。

supplemental_train.csv - 提出期間終了後、このファイルのデータは提出期間中のクリプトアセット価格に置き換わります。評価フェーズでは、訓練、訓練補完、テストセットは、データの欠落を除けば、時間的に連続したものとなる。現在のコピーは、train.csvからおよそ適切な量のデータを満たしただけのもので、プレースホルダとして提供されます。

時系列API詳細
投稿完了までの流れは、時系列紹介ノートを参照ください。時系列APIは前回大会から多少変更になりました!

テストセットには、およそ3ヶ月分のデータが含まれることを想定しています。コンペティションの予測フェーズまでは、APIはトレーニングデータのスライスを配信するだけです。

APIは初期化後に0.5GBのメモリーを必要とします。初期化ステップ(env.iter_test())はそれ以上のメモリを必要としますので、この呼び出しを行うまでモデルをロードしないことをお勧めします。また、APIはデータのロードと配信に30分弱のランタイムを消費します。

APIは、以下の型を使用してデータをロードする。Asset_ID: int8, Count: int32, row_id: int32, Count: int32, Open: float64, High: float64, Low: float64, Close: float64, Volume: float64, VWAP: float64

## 評価指標
平均二乗誤差、R^2、説明分散、相関はいずれも非常に密接な関係にありますが、相関はターゲットと予測の共分散から、次数のボラティリティを正規化する傾向があるという有用な特性を持っています。金融市場（特に暗号市場！）において、ボラティリティの予測はそれ自体が難しい（しかし興味深い！）問題です。相関を指標とすることで、予測問題からノイズを取り除き、より安定した評価指標を提供することが期待されます。

```python
def weighted_correlation(a, b, weights):

  w = np.ravel(weights)
  a = np.ravel(a)
  b = np.ravel(b)

  sum_w = np.sum(w)
  mean_a = np.sum(a * w) / sum_w
  mean_b = np.sum(b * w) / sum_w
  var_a = np.sum(w * np.square(a - mean_a)) / sum_w
  var_b = np.sum(w * np.square(b - mean_b)) / sum_w

  cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
  corr = cov / np.sqrt(var_a * var_b)

  return corr
```

## ターゲット
ターゲットの計算は資産の終値に基づいており、https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition の方法論を使用して提供されたデータから導き出すことができます。関連する関数は以下の通りです。


## 日記
### 20211226
* trainにtestデータも含まれていることに注意！

### 20211229
* PBはほとんど意味がないので、提出は最後の方で考えればいいのでは
* とりあえずローカルで評価指標を見ながら実験をする
* テストも実装した。
* 特徴量
  * ラグ
  * 通貨ごとの集約特徴量
  * optiver参考になりそう

### 20220103
* 自己相関関数
  * https://www.kaggle.com/iamleonie/time-series-interpreting-acf-and-pacf
  * 統計学っぽい話？だと思う。
* 特徴量中和
  * https://www.kaggle.com/yamqwe/g-research-avoid-overfit-feature-neutralization/
  * overfitの際に特徴量とターゲットの間の線形成分を取り除く？
  * 他のコンペやnumeraiなどでも効果が発揮されている。
* 損失関数に何を使うべきか調査
* technical特徴量移動平均などを通貨ごとにしていなかったから意味なくなってる
* テクニカル分析の特徴量を作れるライブラリあるらしい talib


### 20220104
* technical特徴量移動平均などを通貨ごとにしていなかったから意味なくなってる
  * 修正
* lgbmのfevalを設定

### 20220105
* openやcloseのlagは意味があるか
  * 必ずしもcloseと次のopenが一致する訳ではない（そりゃそうか。前の段階の値段で絶対に取引するとは限らない）
  * 意味なくはないかもしれないけど差や割合とかの方が大事そう？
* feature hint
  * シャープレシオ
  * 騰落率
  * close / (60日間移動平均)
* https://www.smbcnikko.co.jp/terms/japan/o/J0735.html
  * オシレーターとは「振り子」や「振り幅」という意味で、投資用語では「買われ過ぎ」や「売られ過ぎ」を示すテクニカル分析手法です。
* 過学習しすぎ。リーク？
  * シャッフルしちゃってた
  * 修正後ある程度の評価が出た
  * 過学習はしてる気がする

### 20220106
* https://www.kaggle.com/lucasmorin/on-line-feature-engineering
  * オンライン特徴量作成
  * testデータでの予測の際にAPIによって分割されたデータが逐一渡されるため、特徴量をその度に作る必要がある。この時に前の情報が必要な特徴量があるときに困る。
  * オンラインで作成する方法を載せてくれているが、これ以外にもかなり作りたい特徴量があるので、多少非効率でも動くならメモリにデータを置いとくなどで対応したい。
* targetについて
  * ターゲットは 15 分間のログリターン（𝑅𝑎）から導き出されます。
  * ログリターンからそれぞれのアセットの情報を引いた感じ？
* lgbm parameters
  * earlystopping: The rule of thumb is to have it at 10% of your num_iterations.
* PurgedGroupTimeSeriesSplitでバリデーション
  * https://www.kaggle.com/marketneutral/purged-time-series-cv-xgboost-optuna


### 20220107
* PurgedGroupTimeSeriesSplitがかなり時間かかる
  * 移動平均などを使用しているため、パージするのは大事そう
  * とりあえずクロスバリデーションはせずに、単純に何日か開けてバリデーションを作る
  * train: 2018-01-01~2021-09-21
  * 特徴量に追加しよう

### 20220108
* submit notebookのサブミットエラーは何かわからん
  * とりあえず特徴量作成の時に残すようにする
  * もしかしたら特徴量がnullにならないものがnullになってて提出時もnullになってしまっているかも


### 20220109
* timeseriesAPIは不正されないように一回しかデータを呼べないようになっているためエラーが出るたびにセッションを再起動する必要がある。。。
* https://www.kaggle.com/yamqwe/purgedgrouptimeseries-cv-with-extra-data-lgbm/notebook#Data-Loading-%F0%9F%97%83%EF%B8%8F
  * validation参考に
* https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/284888
  * 以前のコンペのノートブックまとめ
* https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/285047
  * 使えそうなパッケージやAutoML  
* 毎月最初の１分間が欠落してる
* おそらく提出直前のデータや2018年以前のデータも使わないと勝てない
* https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/294928
  * 特徴量エンジニアリングの参考

### 20220110
* 何日も前の移動平均はそんな必要なさそう（15分後のリターン予測？だから）

### 20220111
* テストのエラーはprintしまくってたから時間かかったのでは？
* 修正してサブミット(lgbm007)
* kfoldのサブミットコード作成
  * mlflowで一つしか保存できていなかった
  * 自動でできるか調査

### 2021012
* テストの総計はおよそ3ヶ月分のデータで、APIの反復はおよそ129600回なので、1ループあたり0.25秒を下回る必要がある
  * modelを毎回中でロードしてた。。。これが原因か？
  * 0.8秒を下回らないといけない。まだ1秒かかる
  * https://www.kaggle.com/lucasmorin/running-algos-fe-for-fast-inference
    * 高速アルゴリズム（移動平均などの）
  * rsiのアルゴリズムがかなり遅かった。

### 20220113
* 特徴量を早く作れるように書き換えて、lgbm学習させたがリークしてる気がする

### 20220114
* https://forum.numer.ai/t/model-diagnostics-feature-exposure/899
  * feature_exposure
  * https://yaakublog.com/numerai-feature-neutralization
* https://note.com/j26/n/n64d9c37167a6（LightGBMのパラメータ)

### 20220115
* row_idではなくindexで行を抜き出していたため、エラーが発生してたっぽい
* なお、timeoutは解消せず、、、

### 20220116
* https://www.sciencedirect.com/science/article/pii/S0169207021001679
  * 論文（ツリー系のモデルの可能性）
* train, sup_train, testは時系列順で連続につながっている！！
* 今後やること
  * 特徴量増やす
  * 高速化は提出できればしないで他の特徴量を試す
  * cvでテスト
  * 学習データ増やす
  * 中和
* 今の市場は強気でそのまま続くなら最近のデータを使って学習して強気市場の特徴を捉えたら勝てる
* purgeをvalidationの前と後ろにとってそれ以外全部trainっていうcvもあった

### 20220117
* 提出できたけどめちゃくちゃスコア低い 
  * moving_averageがnullになっているとか？
  * validationはそこそこ高いのに、、、過学習？

### 20220118
* 開発者のgithubでのコメントを感情分析し、モデルに組み込んで精度が向上した例があるらしい
* Price Rate Of Change (ROC): it measures the percentage change in price between the current price and the price a certain number of periods ago.
* Momentum: it is the rate of acceleration of a security’s price, i.e. the speed at which the price is changing.This measure is particularly useful to identify trends.
* On Balance Volume (OBV ): it is a technical momentum indicator based on the traded volume of an asset to predict changes in stock price.

### 20220119
* テストより前のデータをhistoryに入れておいてる（これが本当に合っているかはわからない）
* ボラティリティの特徴量を追加
  * https://www.kaggle.com/yamqwe/crypto-prediction-volatility-features

### 20220120
* ある銘柄の指標は、他銘柄の指標や自身の過去の値と「比較」することで初めて意味を持ちます。逆に言うと単純に採用した指標には説明力が乏しく、検証でどれだけ良い結果が得られても全く信用するに値しません。 https://we.love-profit.com/entry/2017/01/27/160350
  * 前の時点でのデータとの比較や他の銘柄の比較を行わないと意味ある特徴量にはならないようだ。。。
* ミュータブルなオブジェクトはcopyしないとオブジェクトが同じものを指しているので元の方まで変わっちゃう
  * https://qiita.com/Kaz_K/items/a3d619b9e670e689b6db

### 20220125
* pcaで次元削減
* 

### 20220126
* https://www.youtube.com/watch?v=43zKtrEpsWc&t=3039s
* かなり参考になる
* 定常性を全然考えてなかった・・・
* ベースラインのやつに特徴量入れてみる
* cvも出してみてどうなってるのか確かめる 

### 20220126
* 価格の正規化をしてみる（アセットごとに）
  * cvかなりさがった
* memory削減