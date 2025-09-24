# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。

Model architecture of RNMT+. On the left side, the encoder network has 6 bidirectional LSTM
layers. At the end of each bidirectional layer, the outputs of the forward layer and the backward layer
are concatenated. On the right side, the decoder network has 8 unidirectional LSTM layers, with the first
layer used for obtaining the attention context vector through multi-head additive attention. The attention
context vector is then fed directly into the rest of the decoder layers as well as the softmax layer.
---
RNMT+のモデルアーキテクチャ。左側では、エンコーダーネットワークが6つの双方向LSTM層を持っています。各双方向層の最後で、前方層と後方層の出力が連結されます。右側では、デコーダーネットワークが8つの単方向LSTM層を持ち、最初の層はマルチヘッド加算注意を通じて注意コンテキストベクトルを取得するために使用されます。注意コンテキストベクトルは、残りのデコーダー層およびソフトマックス層に直接供給されます。
---

In this paper, we therefore take a step back and
look at which techniques and methods contribute
significantly to the success of recent architectures,
namely ConvS2S and Transformer, and explore
applying these methods to other architectures, in-
cluding RNMT models. In doing so, we come up
with an enhanced version of RNMT, referred to
as RNMT+, that significantly outperforms all in-
dividual architectures in our setup. We further in-
troduce new architectures built with different com-
ponents borrowed from RNMT+, ConvS2S and
Transformer. In order to ensure a fair setting for
comparison, all architectures were implemented in
the same framework, use the same pre-processed
data and apply no further post-processing as this
may confound bare model performance.
---
この論文では、最近のアーキテクチャ、つまりConvS2SとTransformerの成功に大きく貢献する技術と方法を振り返り、これらの方法を他のアーキテクチャ、特にRNMTモデルに適用することを探ります。その結果、RNMTの強化版であるRNMT+を提案し、これが私たちの設定で全ての個別アーキテクチャを大幅に上回ることがわかりました。さらに、RNMT+、ConvS2S、Transformerから借用した異なるコンポーネントで構築された新しいアーキテクチャを紹介します。比較のための公正な設定を確保するために、すべてのアーキテクチャは同じフレームワークで実装され、同じ前処理されたデータを使用し、さらなる後処理は行いません。これは、モデルのパフォーマンスを混乱させる可能性があるためです。
---

Our contributions are three-fold:
1. In ablation studies, we quantify the effect
of several modeling improvements (includ-
ing multi-head attention and layer normaliza-
tion) as well as optimization techniques (such
as synchronous replica training and label-
smoothing), which are used in recent archi-
tectures. We demonstrate that these tech-
niques are applicable across different model
architectures.
2. Combining these improvements with the
RNMT model, we propose the new RNMT+
model, which significantly outperforms all
fundamental architectures on the widely-used
WMT’14 En→Fr and En→De benchmark
datasets. We provide a detailed model anal-
ysis and comparison of RNMT+, ConvS2S
and Transformer in terms of model quality,
model size, and training and inference speed.
3. Inspired by our understanding of the rela-
tive strengths and weaknesses of individual
model architectures, we propose new model
architectures that combine components from
the RNMT+ and the Transformer model, and
achieve better results than both individual ar-
chitectures.
---
私たちの貢献は三つあります。
1. アブレーション研究において、最近のアーキテクチャで使用されているいくつかのモデリング改善（マルチヘッド注意やレイヤー正規化を含む）および最適化技術（同期レプリカトレーニングやラベルスムージングなど）の効果を定量化します。これらの技術が異なるモデルアーキテクチャに適用可能であることを示します。
2. これらの改善をRNMTモデルと組み合わせて、新しいRNMT+モデルを提案します。これは、広く使用されているWMT’14 En→FrおよびEn→Deベンチマークデータセットで、すべての基本アーキテクチャを大幅に上回ります。RNMT+、ConvS2S、およびTransformerのモデル品質、モデルサイズ、トレーニングと推論速度に関する詳細なモデル分析と比較を提供します。
3. 個々のモデルアーキテクチャの相対的な強みと弱みの理解に触発され、RNMT+とTransformerモデルからコンポーネントを組み合わせた新しいモデルアーキテクチャを提案し、両方の個別アーキテクチャよりも優れた結果を達成します。
---

The newly proposed RNMT+ model architecture
is shown in Figure 1. Here we highlight the key
architectural choices that are different between the
RNMT+ model and the GNMT model. There are
6 bidirectional LSTM layers in the encoder instead
of 1 bidirectional LSTM layer followed by 7 uni-
directional layers as in GNMT. For each bidirec-
tional layer, the outputs of the forward layer and
the backward layer are concatenated before being
fed into the next layer. The decoder network con-
sists of 8 unidirectional LSTM layers similar to the
GNMT model. Residual connections are added to
the third layer and above for both the encoder and
decoder. Inspired by the Transformer model, per-
gate layer normalization (Ba et al., 2016) is ap-
plied within each LSTM cell. Our empirical re-
sults show that layer normalization greatly stabi-
lizes training. No non-linearity is applied to the
LSTM output. A projection layer is added to the
encoder final output.5 Multi-head additive atten-
tion is used instead of the single-head attention in
the GNMT model. Similar to GNMT, we use the bottom decoder layer and the final encoder layer
output after projection for obtaining the recurrent
attention context. In addition to feeding the atten-
tion context to all decoder LSTM layers, we also
feed it to the softmax by concatenating it with the
layer input. This is important for both the quality
of the models with multi-head attention and the
stability of the training process.
Since the encoder network in RNMT+ consists
solely of bi-directional LSTM layers, model par-
allelism is not used during training. We com-
pensate for the resulting longer per-step time with
increased data parallelism (more model replicas),
so that the overall time to reach convergence of
the RNMT+ model is still comparable to that of
GNMT.
We apply the following regularization tech-
niques during training.
---
新しいRNMT+モデルアーキテクチャは図1に示されています。ここでは、RNMT+モデルとGNMTモデルの間で異なる主要なアーキテクチャの選択を強調します。エンコーダーには6つの双方向LSTM層があり、GNMTのように1つの双方向LSTM層の後に7つの単方向層が続くのではありません。各双方向層では、前方層と後方層の出力が次の層に供給される前に連結されます。デコーダーネットワークはGNMTモデルと同様に8つの単方向LSTM層で構成されています。エンコーダーとデコーダーの両方で、3番目の層以降に残差接続が追加されます。Transformerモデルに触発され、ゲート付きレイヤー正規化（Ba et al., 2016）が各LSTMセル内で適用されます。我々の経験的な結果は、レイヤー正規化がトレーニングを大いに安定させることを示しています。LSTM出力には非線形性は適用されません。エンコーダーの最終出力にはプロジェクション層が追加されます。GNMTモデルの単一ヘッド注意の代わりにマルチヘッド加算注意が使用されます。GNMTと同様に、リカレント注意コンテキストを取得するために、ボトムデコーダー層とプロジェクション後の最終エンコーダー層出力を使用します。注意コンテキストをすべてのデコーダーLSTM層に供給するだけでなく、レイヤー入力と連結してソフトマックスにも供給します。これは、マルチヘッド注意を持つモデルの品質とトレーニングプロセスの安定性の両方にとって重要です。
RNMT+のエンコーダーネットワークは双方向LSTM層のみで構成されているため、トレーニング中にモデル並列は使用されません。結果として生じる1ステップあたりの時間の長    さを、データ並列性の向上（より多くのモデルレプリカ）で補います。そのため、RNMT+モデルの収束までの全体的な時間はGNMTと同等です。
トレーニング中に以下の正則化技術を適用します。
---

 Performance comparison. Examples/s are
normalized by the number of GPUs used in the
training job. FLOPs are computed assuming that
source and target sequence length are both 50.
---
パフォーマンス比較。例/秒はトレーニングジョブで使用されるGPUの数で正規化されています。FLOPsは、ソースとターゲットのシーケンス長が両方とも50であると仮定して計算されます。
---

•Dropout: We apply dropout to both embed-
ding layers and each LSTM layer output before
it is added to the next layer’s input. Attention
dropout is also applied.
•Label Smoothing: We use uniform label
smoothing with an uncertainty=0.1 (Szegedy
et al., 2015). Label smoothing was shown to
have a positive impact on both Transformer
and RNMT+ models, especially in the case
of RNMT+ with multi-head attention. Similar
to the observations in (Chorowski and Jaitly,
2016), we found it beneficial to use a larger
beam size (e.g. 16, 20, etc.) during decoding
when models are trained with label smoothing.
•Weight Decay: For the WMT’14 En→De task,
we apply L2 regularization to the weights with
λ = 10−5. Weight decay is only applied to the
En→De task as the corpus is smaller and thus
more regularization is required.
---
•ドロップアウト：埋め込み層とLSTM層の出力にドロップアウトを適用し、次の層の入力に加える前に適用します。注意ドロップアウトも適用されます。
•ラベルスムージング：一様ラベルスムージングを不確実性=0.1（Szegedy et al., 2015）で使用します。ラベルスムージングは、特にマルチヘッド注意を持つRNMT+の場合に、TransformerおよびRNMT+モデルの両方にポジティブな影響を与えることが示されています。（Chorowski and Jaitly, 2016）の観察と同様に、ラベルスムージングでトレーニングされたモデルのデコーディング中に、より大きなビームサイズ（例：16、20など）を使用することが有益であることがわかりました。
•ウェイト減衰：WMT'14 En→Deタスクでは、λ = 10−5の重みをL2正則化に適用します。ウェイト減衰は、コーパスが小さく、したがってより多くの正則化が必要であるため、En→Deタスクにのみ適用されます。
---

Additional projection aims to reduce the dimensionality
of the encoder output representations to match the decoder
stack dimension.
---
追加のプロジェクションは、エンコーダー出力表現の次元数をデコーダースタックの次元に合わせることを目的としています。
---