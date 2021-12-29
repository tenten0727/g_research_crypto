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
