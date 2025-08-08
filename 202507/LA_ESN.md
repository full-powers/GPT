# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。
# 
# 簡単な使用方法: 
# 1. 英文を入力する
# 2. 英文の下で「ja」と入力してTabキーを押す → 翻訳用区切りが挿入される
# 3. 日本語翻訳を入力する
#
# その他の便利機能:
# - 「trans」+ Tab: 英文と翻訳のテンプレート
# - 「new」+ Tab: 新しい英文翻訳ペアを追加  
# - Ctrl+Shift+J: 翻訳用区切り「---」を挿入
# - Ctrl+Shift+N: 新しい翻訳ペアを挿入

---
Abstract: Time-series data is an appealing study topic in data mining and has a broad range of
applications. Many approaches have been employed to handle time series classification (TSC) challenges with promising results, among which deep neural network methods have become mainstream.
Echo State Networks (ESN) and Convolutional Neural Networks (CNN) are commonly utilized
as deep neural network methods in TSC research. However, ESN and CNN can only extract local
dependencies relations of time series, resulting in long-term temporal data dependence needing to
be more challenging to capture. As a result, an encoder and decoder architecture named LA-ESN is
proposed for TSC tasks. In LA-ESN, the encoder is composed of ESN, which is utilized to obtain the
time series matrix representation. Meanwhile, the decoder consists of a one-dimensional CNN (1D
CNN), a Long Short-Term Memory network (LSTM) and an Attention Mechanism (AM), which can
extract local information and global dependencies from the representation. Finally, many comparative
experimental studies were conducted on 128 univariate datasets from different domains, and three
evaluation metrics including classification accuracy, mean error and mean rank were exploited to
evaluate the performance. In comparison to other approaches, LA-ESN produced good results.
---
要約: 時系列データはデータマイニングにおいて魅力的な研究テーマであり、幅広い応用があります。多くのアプローチが時系列分類（TSC）の課題に対処するために採用され、有望な結果を得ています。その中でも、深層ニューラルネットワーク手法が主流となっています。Echo State Networks（ESN）と畳み込みニューラルネットワーク（CNN）は、TSC研究で一般的に使用される深層ニューラルネットワーク手法です。しかし、ESNとCNNは時系列の局所的な依存関係のみを抽出できるため、長期的な時間データの依存関係を捉えることが難しいという問題があります。その結果、LA-ESNと呼ばれるエンコーダーとデコーダーのアーキテクチャがTSCタスクのために提案されました。LA-ESNでは、エンコーダーはESNで構成され、時系列の行列表現を取得するために使用されます。一方、デコーダーは1次元CNN（1D CNN）、長短期記憶ネットワーク（LSTM）、およびアテンションメカニズム（AM）で構成され、局所情報とグローバルな依存関係を表現から抽出できます。最後に、異なるドメインからの128の単変量データセットに対して多くの比較実験が行われ、分類精度、平均誤差、平均ランクの3つの評価指標が性能評価に利用されました。他のアプローチと比較して、LA-ESNは良好な結果を出しました。
---

Massive data resources have accumulated in numerous industries in the quickly
increasing information era, and large-scale data provide valuable content. Time series
data is a set of data points arranged in chronological order, which can be divided into two
categories: univariate and multivariate. In univariate time series data, only one variable
varies over time, while in multivariate time series data, multiple variables change over
time [1]. Time series data are now widely used in a wide range of applications, including
statistics, pattern recognition, earthquake prediction, econometrics, astronomy, signal
processing, control engineering, communication engineering, finance, weather forecasting,
the Internet of Things, and medical care [2].
Much work has been completed around the TSC problem in the last two decades. This
enlightened and attractive data mining topic focuses on classifying data points indexed
by time by predicting their labels. Time series classification is an essential task in data
mining and has been extensively employed in many fields. For example, in the field of
network traffic, Xiao et al. [3] proposed traffic classification for dynamic network flows. In
medical diagnosis, R. Michida et al. [4] proposed a deep learning method for classifying
lesions based on JNET classification for computer-aided diagnosis systems for colorectal
magnification NBI endoscopy. In finance, Mori et al. [5] and Wan et al. [6] proposed methods
for the early classification of time series using multi-objective optimization techniques and
financial strategies for classifying chart patterns in time series.
---
Massiveのデータリソースが急速に増加する情報時代において、さまざまな産業に蓄積されています。大規模なデータは貴重なコンテンツを提供します。時系列データは、時間的に配置されたデータポイントのセットであり、単変量と多変量の2つのカテゴリに分けられます。単変量時系列データでは、1つの変数のみが時間とともに変化しますが、多変量時系列データでは、複数の変数が時間とともに変化します。[1]。時系列データは、統計、パターン認識、地震予測、計量経済学、天文学、信号処理、制御工学、通信工学、金融、天気予報、モノのインターネット、医療など、幅広い応用で広く使用されています[2]。
過去20年間で、TSC問題に関する多くの研究が行われてきました。この魅力的なデータマイニングのトピックは、時間でインデックス付けされたデータポイントを分類し、それらのラベルを予測することに焦点を当てています。時系列分類はデータマイニングにおいて重要なタスクであり、多くの分野で広く利用されています。例えば、ネットワークトラフィックの分野では、Xiaoら[3]は動的ネットワークフローのトラフィック分類を提案しました。医療診断では、R. Michidaら[4]は、結腸直腸拡大NBI内視鏡のコンピュータ支援診断システムのためのJNET分類に基づく病変分類のための深層学習法を提案しました。金融分野では、Moriら[5]とWanら[6]は、マルチオブジェクティブ最適化技術を使用した時系列の早期分類方法と、時系列のチャートパターンを分類するための金融戦略を提案しました。
---

In light of the preceding work, we propose an end-to-end TSC model called LA-ESN
for time series classification. The LA-ESN consists of an echo memory encoder and a
decoder. The reservoir layer makes up the echo memory encoder, while the decoder is
made up of a 1D CNN, an LSTM and Attention. The storage layer first projects the time
series into a high-dimensional nonlinear space to generate echo states step by step. The
echo memory matrix is formed by collecting the echo states of all time steps in chronological
order. To capture the critical historical information in the time series, we design a decoder
that applies a 1D CNN, an LSTM and Attention to the echo memory matrix, respectively.
To increase network efficiency, the 1D CNN and LSTM are employed to extract multi-scale
features and retrieves global information from the echo memory matrix. Then, the Attention
is used to extract the essential information from the global information. LA-ESN is an
acceptable classification approach, according to experimental findings on a variety of time
series datasets.
---
前述の研究を踏まえ、時系列分類のためのエンドツーエンドのTSCモデルであるLA-ESNを提案します。LA-ESNは、エコーメモリエンコーダーとデコーダーで構成されています。リザーバ層はエコーメモリエンコーダーを形成し、デコーダーは1次元CNN、LSTM、およびアテンションで構成されています。ストレージ層は、時系列を高次元の非線形空間に投影し、エコーステートを段階的に生成します。エコーメモリーマトリックスは、すべての時間ステップのエコーステートを時間順に収集することで形成されます。時系列内の重要な履歴情報を捉えるために、デコーダーを設計し、1次元CNN、LSTM、およびアテンションをそれぞれエコーメモリーマトリックスに適用します。ネットワーク効率を向上させるために、1次元CNNとLSTMはマルチスケール特徴を抽出し、エコーメモリーマトリックスからグローバル情報を取得します。その後、アテンションを使用してグローバル情報から重要な情報を抽出します。LA-ESNは、さまざまな時系列データセットに関する実験結果によれば、受け入れ可能な分類アプローチです。
---

## 翻訳テンプレートの使用方法（改良版）

### 最も簡単な方法:
1. 英文を入力
2. 英文の下で `ja` + Tab
3. 日本語翻訳を入力

### その他のスニペット:
- `trans` + Tab → 英文と翻訳のテンプレート
- `new` + Tab → 新しい翻訳ペアを追加

### キーボードショートカット:
- `Ctrl+Shift+J` → 翻訳用区切り「---」を挿入
- `Ctrl+Shift+N` → 新しい翻訳ペアを挿入

### 使用例:
英文を入力後、`ja`と入力してTabキーを押すと、下記のように自動で区切りが挿入されます：

```
Your English text here.
---
ここに日本語翻訳を入力
---
```

<!-- 新しい英文を追加する場合は、この下に入力してください -->

The following are the main contributions of this paper:
(1) We propose a simple end-to-end model LA-ESN for handling time series classification tasks;
(2) We modify the output layer of ESN to handle time series better and use CNN and
LSTM as output layers to finish feature extraction;
(3) The attention mechanism is deployed behind both CNN and LSTM, which effectively
improves the effectiveness and computing efficiency of LA-ESN;
(4) Experiments on various time series datasets show that LA-ESN is efficacious.
The rest of this paper is organized as follows. Section 2 reviews the related work of this study. Section 3 explains the proposed method. Section 4 provides a description
of the UCR time series database and the results of the comparison experiments. A brief
conclusion is given in Section 5.
---
この論文の主な貢献は以下の通りです：
(1) 時系列分類タスクを処理するためのシンプルなエンドツーエンドモデルLA-ESNを提案しています；
(2) 時系列をより良く処理するためにESNの出力層を修正し、特徴抽出を完了するためにCNNとLSTMを出力層として使用しています；
(3) アテンションメカニズムはCNNとLSTMの両方の後ろに配置され、LA-ESNの有効性と計算効率を効果的に向上させています；
(4) さまざまな時系列データセットでの実験により、LA-ESNが有効であることが示されています。
この論文の残りの部分は以下のように構成されています。第2章では、この研究の関連研究をレビューします。第3章では、提案された手法について説明します。第4章では、UCR時系列データベースの説明と比較実験の結果を提供します。第5章では簡潔な結論を述べます。
---

The framework of LA-ESN is divided into two parts: encoding and decoding. In
the encoding part, each frame of the input time series is mapped into a high-dimensional
reservoir state space to obtain an Echo State Representation (ESR). The ESRs of all time
steps are stored in a memory matrix. We simultaneously design two ways to decode the
memory matrix in the decoding phase. First, we adopt a Multi-scale 1D CNN [31–33] to
extract the local information. Second, utilizing LSTM and Attention, long-term dependent
information is recovered from the memory matrix. Both attention mechanisms are intended
to reduce irrelevant information while increasing computational efficiency. Finally, the
local and long-term dependent information obtained by the two methods is pooled and
merged, and then the merged features are passed through a fully connected layer. The
conditional probability distribution of the categories is calculated using a Softmax layer.
The proposed LA-ESN model’s general design is seen in Figure 1.
---
提案されたLA-ESNモデルの一般的な設計は、図1に示されています。
LA-ESNのフレームワークは、エンコードとデコードの2つの部分に分かれています。エンコード部分では、入力時系列の各フレームが高次元のリザーバ状態空間にマッピングされ、エコーステート表現（ESR）を取得します。すべての時間ステップのESRはメモリマトリックスに保存されます。デコードフェーズでは、メモリマトリックスをデコードするために2つの方法を同時に設計しています。まず、マルチスケール1D CNN [31–33]を採用して局所情報を抽出します。次に、LSTMとアテンションを利用して、メモリマトリックスから長期依存情報を回復します。両方のアテンションメカニズムは、無関係な情報を減らしながら計算効率を向上させることを目的としています。最後に、2つの方法で得られた局所および長期依存情報がプールされて統合され、その後統合された特徴が全結合層を通過します。Softmax層を使用してカテゴリの条件付き確率分布が計算されます。
---

The primitive ESN includes an input, reserve pool, and output layer. The diagram
of the primitive ESN model is shown in Figure 2. The primitive ESN uses a reserve pool
of randomly sparsely connected neurons as the hidden layer of a high-dimensional and
nonlinear input representation. The weights of the hidden layer of the ESN are generated in advance rather than by training. Meanwhile, they are trained from the hidden layer to the output layer. Therefore, the generated reserve pool has some good properties
that guarantee excellent performance by using only linear methods to train the weights
from the reserve pool to the output layer. Given a k-dimensional input, i(t) with time
step t, the state of the reserve pool with time step t − 1 is r(t - 1).
---
原始のESNは、入力、リザーブプール、および出力層で構成されています。原始ESNモデルの図は図2に示されています。原始ESNは、ランダムに疎結合されたニューロンのリザーブプールを高次元かつ非線形の入力表現の隠れ層として使用します。ESNの隠れ層の重みは、事前に生成され、トレーニングによってではなく、隠れ層から出力層へのトレーニングによって生成されます。そのため、生成されたリザーブプールは、リザーブプールから出力層への重みを線形手法のみでトレーニングすることで優れた性能を保証するいくつかの良好な特性を持っています。k次元の入力i(t)が時間ステップtで与えられた場合、時間ステップt - 1でのリザーブプールの状態はr(t - 1)です。
---

In previous studies, multi-scale convolution has been used as a feature extractor to
extract effective classification features from time series representations. Alternatively, LSTM
is used to learn straightforwardly from the input time series. Both approaches are capable
of classifying time series to some extent, although they might be improved. Therefore, we
propose to appropriately adapt and then combine them to be used as the output layer of
ESN to learn better feature information from the echo states.
---
以前の研究では、マルチスケール畳み込みが特徴抽出器として使用され、時系列表現から効果的な分類特徴を抽出しています。また、LSTMは入力時系列から直接学習するために使用されます。どちらのアプローチもある程度時系列を分類することができますが、改善の余地があります。したがって、ESNの出力層として使用するために、それらを適切に適応させて組み合わせることを提案します。
---

On the one hand, we employ multiple scales of 1D CNNs for convolution operations
along the time direction and use multiple filters for each time scale. The batch normalization
operation follows the convolution operations to avoid the gradient disappearance problem.
Next, ReLU as a correction layer for the activation function is adopted, which can improve
the nonlinear and sparsity connection between the levels to reduce over-fitting. Therefore,
batch normalization and ReLU can achieve more robust learning. Finally, the multi-scale
features are concatenated into a new feature. The structure diagram of a multi-scale 1D
convolution is shown in Figure 3.
---
一方で、時間方向に沿った畳み込み操作のために複数のスケールの1D CNNを使用し、各時間スケールに対して複数のフィルターを使用します。バッチ正規化操作は、勾配消失問題を回避するために畳み込み操作の後に行われます。次に、ReLUを活性化関数の補正層として採用し、レベル間の非線形性と疎結合を改善して過学習を減少させます。したがって、バッチ正規化とReLUはより堅牢な学習を実現できます。最後に、マルチスケール特徴が新しい特徴として連結されます。マルチスケール1D畳み込みの構造図は図3に示されています。
---

On the other hand, LSTM is extremely successful at dealing with time series chalenges. Therefore, we propose to learn the long-term global dependence between thstates from the echo state matrix using LSTM to make LA-ESN learn more robust classfication feature information. In LA-ESN, the LSTM receives the echo state matrix asmultivariate state matrix with a single time step, and the operations of the LSTM are shown as follows:
---
一方で、LSTMは時系列の課題に非常に成功しています。したがって、LA-ESNがより堅牢な分類特徴情報を学習できるように、LSTMを使用してエコーステートマトリックスから状態間の長期的なグローバル依存関係を学習することを提案します。LA-ESNでは、LSTMはエコーステートマトリックスを単一の時間ステップを持つ多変量状態マトリックスとして受け取り、LSTMの操作は以下のように示されます： 
---

3. The diagram of a multi-scale 1D convolution. The stride of each 1D convolution LA-ESN is set to 1. Different colours represent the convolution results with different convolution kernel sizes. From top to bottom, the convolution results are shown for convolution kernel sizes 3 and 8, respectively.
---
3. マルチスケール1D畳み込みの図。各1D畳み込みのストライドは1に設定されています。異なる色は、異なる畳み込みカーネルサイズでの畳み込み結果を表しています。上から下へ、畳み込みカーネルサイズ3と8の畳み込み結果がそれぞれ示されています。
---

Then, the attention module is integrated, which is commonly used in natural language
processing. The context vector ci depends on a sequence of annotations (hi, · · · , hTx). Each
annotation hi contains information about the entire input sequence, focusing on the part of
the input sequence around the i-th word. The encoder maps the input sequence to a new
sequence. The context vector ci
is a weighted sum of these annotations as below:
---
次に、自然言語処理で一般的に使用されるアテンションモジュールが統合されます。コンテキストベクトルciは、アノテーションのシーケンス（hi, · · · , hTx）に依存します。各アノテーションhiは、i番目の単語の周りの入力シーケンスの部分に焦点を当て、入力シーケンス全体に関する情報を含んでいます。エンコーダーは入力シーケンスを新しいシーケンスにマッピングします。コンテキストベクトルciは、以下のようにこれらのアノテーションの重み付き和です：
---


There are two advantages to using the AM. First, AM can apply various weights to distinct
echo state representations of the same time step. In other words, AM can give more weight
to information that is more significant for categorization while suppressing irrelevant data.
Second, including the AM increases the model’s running speed.
---
AMを使用することには2つの利点があります。第一に、AMは同じ時間ステップの異なるエコーステート表現に異なる重みを適用できます。言い換えれば、AMは分類にとってより重要な情報により多くの重みを与え、無関係なデータを抑制できます。第二に、AMを含めることでモデルの実行速度が向上します。
---
This study proposes a deep learning model called LA-ESN for the end-to-end classification of univariate time series. The original time series is first passed through the ESN
model in LA-ESN to obtain the echo state representation matrix. Then, the echo state representation matrix is used as the input of the LSTM module and the multi-scale convolutional
module, and feature extraction operations are performed on them. Finally, the results from
the two modules are concatenated, and the softmax function is utilized to produce the
final classification results. The attention-based LSTM module can automatically record
the long-term time dependence of the sequences, and the multi-scale 1D convolutional
attention module can extract feature information from echo state representations at different scales, highlighting the spatial sparsity and heterogeneity of the data. Without any
data reshaping or pre-processing, the model performs better. Based on extensive trials
on the UCR time series dataset, the proposed LA-ESN model outperforms several older
approaches and current popular deep learning methods in the great majority of selected
datasets. Experiments suggest that our technique may perform better on certain datasets.
However, this experiment still needs to be improved in studying the class imbalance
problem. As a consequence, in the future, we will improve the model to obtain the best
results on most datasets while also addressing the class imbalance issue. Second, we
can consider modifying the model to accomplish the classification of multivariate time
series data. Finally, LA-ESN is a more generic model in the field of time series, and we can
subsequently consider using LA-ESN for tasks such as time series prediction and clustering.
---
この研究では、単変量時系列のエンドツーエンド分類のためにLA-ESNと呼ばれる深層学習モデルを提案します。元の時系列は最初にLA-ESNのESNモデルを通過してエコーステート表現行列を取得します。次に、エコーステート表現行列はLSTMモジュールとマルチスケール畳み込みモジュールの入力として使用され、それらに対して特徴抽出操作が実行されます。最後に、2つのモジュールからの結果が連結され、ソフトマックス関数が利用されて最終的な分類結果が生成されます。アテンションベースのLSTMモジュールはシーケンスの長期的な時間依存性を自動的に記録でき、マルチスケール1D畳み込みアテンションモジュールは異なるスケールでのエコーステート表現から特徴情報を抽出でき、データの空間的スパース性と異質性を強調します。データの再形成や前処理を行わなくても、モデルの性能は向上します。UCR時系列データセットでの広範な試行に基づいて、提案されたLA-ESNモデルは、選択されたデータセットの大多数において、いくつかの古いアプローチや現在の人気のある深層学習手法を上回っています。実験結果は、我々の手法が特定のデータセットでより良い性能を発揮する可能性があることを示唆しています。
しかし、この実験は依然としてクラス不均衡問題の研究において改善が必要です。その結果、今後はほとんどのデータセットで最良の結果を得るためにモデルを改善し、クラス不均衡の問題にも対処します。第二に、マルチ変量時系列データの分類を達成するためにモデルを修正することを検討できます。最後に、LA-ESNは時系列の分野でより一般的なモデルであり、我々はその後、LA-ESNを時系列予測やクラスタリングなどのタスクに使用することを検討できます。
---

The average accuracy and standard deviation of our proposed LA-ESN method five
times and other conventional TSC classifiers on 76 datasets are shown in Table 2. As seen
from Table 2, we can conclude that the following: (1) LA-ESN achieves higher classification
accuracy than the other 12 traditional methods on 36 datasets and achieves comparable
results on the other 25 datasets. The total number significantly outperforms other methods.
(2) The average classification accuracy of all datasets of the proposed LA-ESN is also
superior to the other compared methods. (3) For MR, the performance of LA-ESN is only
slightly worse than that of the FCOTE method but significantly better than that of the other
methods. (4) The ME of the proposed LA-ESN is slightly higher than that of the FCOTE
and ST methods but lower than that of the other methods. Therefore, the experimental
results indicate that LA-ESN is effective for time series classification in most cases.
---
提案されたLA-ESN手法の平均精度と標準偏差を他の従来のTSC分類器と比較した結果を、76のデータセットで5回実行した結果を表2に示します。表2からわかるように、以下のことが結論できます：(1) LA-ESNは、36のデータセットで他の12の従来の手法よりも高い分類精度を達成し、他の25のデータセットでも同等の結果を達成しています。総数は他の手法を大幅に上回っています。(2) 提案されたLA-ESNのすべてのデータセットの平均分類精度も、他の比較手法よりも優れています。(3) MRに関しては、LA-ESNの性能はFCOTE手法よりもわずかに劣りますが、他の手法よりも大幅に優れています。(4) 提案されたLA-ESNのMEはFCOTEおよびST手法よりもわずかに高いですが、他の手法よりは低いです。したがって、実験結果は、LA-ESNがほとんどの場合で時系列分類に効果的であることを示しています。
---

To better compare multiple classifiers on multiple datasets, we applied the pairwise
post hoc analysis proposed by Benavoli et al. [39], where mean rank comparisons were
computed using the Wilcoxon signature rank test [40] corrected for Holm’s alpha (5%)
[41]. A critical difference diagram [42] is used to depict the outcome. The more right the
classifier is, the better it is. As illustrated in Figure 6, the classifiers connected by thick
lines reflect no significant difference in accuracy between the two classifiers. Figure 6
shows the pairwise statistical difference comparison between LA-ESN and traditional
classifiers and traditional classifiers. As can be seen from Figure 6, our model is located to
the right of the majority of approaches.
---
複数のデータセットで複数の分類器を比較するために、Benavoliら[39]によって提案されたペアワイズポストホック分析を適用しました。この分析では、Wilcoxon署名ランクテスト[40]を使用して平均ランクの比較が行われ、Holmのアルファ（5%）[41]で補正されました。結果を示すために、クリティカルディファレンス図[42]が使用されます。分類器が右に位置するほど、分類器の性能が良いことを示します。図6に示されているように、太い線で結ばれた分類器は、2つの分類器間で精度に有意な差がないことを反映しています。図6は、LA-ESNと従来の分類器とのペアワイズ統計的差異比較を示しています。図6からわかるように、我々のモデルは、ほとんどのアプローチの右側に位置しています。
---

From Table 3, we can obtain the following: (1) On 64 datasets, LA-ESN has the highest
classification accuracy. Furthermore, the total number significantly outperforms other
methods. (2) The average classification accuracy of all datasets of the proposed LA-ESN
is also superior to other compared methods. (3) The MR value of the proposed LA-ESN
is 2.430, which is higher than the Inception Time method but smaller than the other three
classifiers. (4) The proposed LA-ESN has a ME value of 0.047, which is lower than the MLP
and FCN but greater than ResNet and Inception Time. (5) The performances of LA-ESN and
Inception Time perform better than the other three methods. This reason is that LA-ESN
and Inception Time can efficiently handle temporal dependencies in time series with high
nonlinear mapping capacity and dynamic memory. (6) When the epoch is set as 500, the
running time of the proposed LA-ESN on small datasets is concise while is acceptable
on large datasets. A paired statistical difference comparison of the five deep learning
methods is shown in Figure 7. As seen in the diagram, our model is to the right of the
majority of methods.
---
表3から、以下のことがわかります：(1) 64のデータセットで、LA-ESNは最高の分類精度を持っています。さらに、総数は他の手法を大幅に上回っています。(2) 提案されたLA-ESNのすべてのデータセットの平均分類精度も他の比較手法よりも優れています。(3) 提案されたLA-ESNのMR値は2.430で、Inception Time手法よりも高いですが、他の3つの分類器よりは小さいです。(4) 提案されたLA-ESNのME値は0.047で、MLPおよびFCNよりも低いですが、ResNetおよびInception Timeよりは大きいです。(5) LA-ESNとInception Timeの性能は他の3つの手法よりも優れています。この理由は、LA-ESNとInception Timeが高い非線形マッピング能力と動的メモリを持つ時系列の時間依存性を効率的に処理できるためです。(6) エポックを500に設定した場合、提案されたLA-ESNの小規模データセットでの実行時間は短く、大規模データセットでは許容範囲内です。5つの深層学習手法のペアワイズ統計的差異比較を図7に示します。図からわかるように、我々のモデルはほとんどの手法の右側に位置しています。
---

In this subsection, a step-by-step exploration has been constructed further to verify
the validity of the proposed LA-ESN model. We divided the model into four modules:
ESN-LSTM, ESN-CNN, ESN_LSTM_ATT (ELA) and ESN_CNN_ATT (ECA), and conducted experiments on 17 datasets. These four methods first employ the ESN to perform
representation learning on time series and then use different classifiers such as LSTM,
CNN, LSTM with attention (LSTM_ATT) and CNN with attention (CNN_ATT) to extract
feature information from the representation. In order to verify the validity of each module, a unified setting is adopted for the parameters. In ESN, the spectral radius SR is 0.9,
the input cell scale IS is 0.1, the reservoir sparsity SP is 0.7, and the size of the reservoir
is 32. The epoch of the whole experiment is 500, and the batch size is 25. LSTM_ATT can
automatically record long-term temporal dependencies of sequences, and CNN_ATT can
extract feature information from different scales. LA-ESN combines both advantages to
extract more valuable information from datasets and achieve better classification accuracy.
Table 4 displays the outcomes of LA-ESN and four distinct classifiers on 17 datasets. On
13 datasets, it can be seen that LA-ESN produces the best results. At the same time, we
build the critical difference diagram shown in Figure 8.
---
このサブセクションでは、提案されたLA-ESNモデルの有効性を検証するために、段階的な探索をさらに構築しました。モデルを4つのモジュールに分割しました：ESN-LSTM、ESN-CNN、ESN_LSTM_ATT（ELA）、およびESN_CNN_ATT（ECA）で、17のデータセットで実験を行いました。これらの4つの手法は、最初にESNを使用して時系列の表現学習を行い、その後、LSTM、CNN、アテンション付きLSTM（LSTM_ATT）、およびアテンション付きCNN（CNN_ATT）などの異なる分類器を使用して、表現から特徴情報を抽出します。各モジュールの有効性を検証するために、パラメータには統一された設定が採用されます。ESNでは、スペクトル半径SRは0.9、入力セルスケールISは0.1、リザーバのスパース性SPは0.7、リザーバのサイズは32です。実験全体のエポックは500で、バッチサイズは25です。LSTM_ATTはシーケンスの長期的な時間依存性を自動的に記録でき、CNN_ATTは異なるスケールから特徴情報を抽出できます。LA-ESNは両方の利点を組み合わせて、データセットからより価値のある情報を抽出し、より良い分類精度を達成します。表4には、LA-ESNと4つの異なる分類器の17のデータセットでの結果が表示されています。13のデータセットで、LA-ESNが最良の結果を出していることがわかります。同時に、図8に示されている重要な差異図を構築します。
---

The term rm demonstrates as the m-th column of R, expressing the echo state of the m-th time sequence at all time steps.
---
The term rmは、Rのm番目の列として表され、すべての時間ステップでのm番目の時系列のエコーステートを表します。
---

The smaller average of MR may indicate that the model is more accurate than the
other methods on most datasets.
---
MRの平均値が小さいほど、モデルがほとんどのデータセットで他の手法よりも正確であることを示す可能性があります。
---