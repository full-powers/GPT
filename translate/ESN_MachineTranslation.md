# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。

We present neural machine translation (NMT)
models inspired by echo state network (ESN),
named Echo State NMT (ESNMT), in which
the encoder and decoder layer weights are randomly generated then fixed throughout training. We show that even with this extremely
simple model construction and training procedure, ESNMT can already reach 70-80%
quality of fully trainable baselines. We examine how spectral radius of the reservoir,
a key quantity that characterizes the model,
determines the model behavior. Our findings indicate that randomized networks can
work well even for complicated sequence-to-sequence prediction NLP tasks.
---
我々は、エコー状態ネットワーク（ESN）に触発されたニューラル機械翻訳（NMT）モデル、エコー状態NMT（ESNMT）を提案する。このモデルでは、エンコーダーとデコーダーの層の重みはランダムに生成され、トレーニング中は固定される。非常にシンプルなモデル構造とトレーニング手順であっても、ESNMTは完全に学習可能なベースラインの70-80%の品質に達することができることを示す。リザーバーのスペクトル半径という、モデルを特徴付ける重要な量がモデルの挙動をどのように決定するかを調べる。我々の発見は、ランダム化されたネットワークが複雑なシーケンスツーシーケンス予測NLPタスクでもうまく機能する可能性があることを示している。
---

In this paper we propose a neural machine translation (NMT) model in which the encoder and decoder layers are randomly generated then fixed
throughout training. We show that despite the extreme simplicity in model construction and training
procedure, the model still performs surprisingly
well, reaching 70-80% BLEU scores given by a
fully trainable model of the same architecture.
Our proposal is inspired by Echo State Network
(ESN) (Jaeger, 2001; Maass et al., 2002), a special
type of recurrent neural network (RNN) whose recurrent and input matrices are randomly generated
and untrained. Such a model building procedure
is counter-intuitive, however as long as its dynamical behavior (characterized by a few key model
hyperparameters) properly approximates the underlying dynamics of a given sequence processing
task, randomized models can also yield competitive
performance. If we view language processing from
a dynamical system’s perspective (Elman, 1995),
ESN can be an effective model for NLP tasks as
well.
---
この論文では、エンコーダーとデコーダーの層がランダムに生成され、トレーニング中は固定されるニューラル機械翻訳（NMT）モデルを提案する。モデル構造とトレーニング手順の極端なシンプルさにもかかわらず、同じアーキテクチャの完全に学習可能なモデルが与えるBLEUスコアの70-80%に達することを示す。我々の提案は、エコー状態ネットワーク（ESN）（Jaeger, 2001; Maass et al., 2002）に触発されている。これは、リカレントおよび入力行列がランダムに生成され、学習されない特殊なタイプのリカレントニューラルネットワーク（RNN）である。このようなモデル構築手順は直感に反するが、その動的挙動（いくつかの主要なモデルハイパーパラメータによって特徴付けられる）が特定のシーケンス処理タスクの基礎となる動力学を適切に近似する限り、ランダム化されたモデルも競争力のある性能を発揮できる。言語処理を動的システムの観点から見るならば（Elman, 1995）、ESNはNLPタスクにも効果的なモデルとなり得る。
---

There are existing works that apply randomized
approaches similar to ESN to NLP tasks (Tong
et al., 2007; Hinaut and Dominey, 2012; Wieting
and Kiela, 2019; Enguehard et al., 2019), which
report the effectiveness of using representations
produced by random encoders. However the capability of ESN in directly handling more general
and complicated sequence-to-sequence (seq2seq)
prediction tasks has not been investigated yet.
---
既存の研究では、ESNに類似したランダム化アプローチをNLPタスクに適用している（Tong et al., 2007; Hinaut and Dominey, 2012; Wieting and Kiela, 2019; Enguehard et al., 2019）。これらの研究は、ランダムエンコーダーによって生成された表現を使用する効果を報告している。しかし、より一般的で複雑なシーケンスツーシーケンス（seq2seq）予測タスクを直接処理するESNの能力はまだ調査されていない。
---

We propose an Echo State NMT
model with a randomized encoder and decoder, extending ESN to a challenging seq2seq prediction
task, and study its uncharacteristic effectiveness
in MT. This also provides an interesting opportunity for model compression, as one only needs to
store one single random seed offline, from which
all randomized model components can be deterministically recovered.
---
ランダム化されたエンコーダーとデコーダーを持つエコー状態NMTモデルを提案し、ESNを挑戦的なseq2seq予測タスクに拡張し、MTにおけるその非典型的な効果を研究する。これは、モデル圧縮のための興味深い機会も提供する。なぜなら、オフラインで1つのランダムシードを保存するだけで、すべてのランダム化されたモデルコンポーネントを決定論的に再現できるからである。
---

Inspired by the simple yet effective construction
of ESN, we are interested in extending ESN to
challenging sequence-to-sequence prediction tasks,
especially NMT. We propose an ESN-based NMT
model whose architecture follows RNMT+ (Chen
et al., 2018), the state-of-the-art RNN-based NMT model. Unlike RNMT+ which is fully trainable, we
simply replace all recurrent layers in the encoder
and decoder with echo state layers as shown in
Eq. 1, and call this model ESNMT.
In addition to the simple RNN cell employed
by the original ESN (Eq. 1), we also explore a
variation of ESNMT which employs the LSTM
cell (Hochreiter and Schmidhuber, 1997). That is,
we randomly generate all weight matrices in the
LSTM and keep them fixed. We call this version
ESNMT-LSTM.
In the models above, the trainable components
are word embedding, softmax and attention layers.
Instead of freezing both encoder and decoder, we
also investigate settings where only the encoder
or decoder is frozen. We further consider cases
where even the attention and embedding layers are
randomized and fixed. These variations of architectures are compared in Sec.4.2.
We note that the size of the reservoir can be
cheaply increased since they do not need to be
trained, which often leads to better performance.
We nevertheless constrain the ESNMT model size
to be the same as trainable baselines in our experiments, even though the latter contain way more
trainable parameters.
---
ESNのシンプルでありながら効果的な構造に触発され、我々はESNを挑戦的なシーケンスツーシーケンス予測タスク、特にNMTに拡張することに興味を持つ。RNMT+（Chen et al., 2018）に従ったアーキテクチャを持つESNベースのNMTモデルを提案する。RNMT+は完全に学習可能であるのに対し、我々はエンコーダーとデコーダーのすべてのリカレント層をエコー状態層に置き換え（Eq. 1参照）、このモデルをESNMTと呼ぶ。
元のESN（Eq. 1）で使用されるシンプルなRNNセルに加えて、LSTMセル（Hochreiter and Schmidhuber, 1997）を使用するESNMTのバリエーションも探る。つまり、
LSTMのすべての重み行列をランダムに生成し、それらを固定する。このバージョンをESNMT-LSTMと呼ぶ。
上記のモデルでは、学習可能なコンポーネントは単語埋め込み、ソフトマックス、およびアテンション層である。エンコーダーとデコーダーの両方を固定する代わりに、エンコーダーまたはデコーダーのみを固定する設定も調査する。
さらに、アテンション層と埋め込み層さえもランダム化して固定する場合も考慮する。これらのアーキテクチャのバリエーションは、Sec.4.2で比較される。
リザーバーのサイズは、トレーニングを必要としないため、安価に増やすことができ、これがしばしばより良い性能につながることに注意する。しかしながら、実験ではESNMTモデルのサイズを学習可能なベースラインと同じに制約する。後者ははるかに多くの学習可能なパラメータを含んでいるにもかかわらず。
---

As described in Sec.2, two critical hyperparameters
that determine the dynamics of ESN and its performance are the spectral norm of the reservoir matrix
and input scale. While common practice manually
tunes these hyperparameters for specific tasks, we
treat them as trainable parameters and let the training procedure find suitable values. Specifically, we
modify the ESN layer in Eq. 1.


where ρl
and γl
are learnable scaling factors for the
reservoir of the l
th layer and input transformation
matrices respectively. Similar modification is applied to the LSTM state transition formulation in
ESNMT-LSTM.
---
前述のように、ESNの動力学とその性能を決定する2つの重要なハイパーパラメータは、リザーバー行列のスペクトルノルムと入力スケールである。一般的な慣習では、これらのハイパーパラメータを特定のタスクに手動で調整するが、我々はそれらを学習可能なパラメータとして扱い、トレーニング手順に適切な値を見つけさせる。具体的には、Eq. 1のESN層を次のように修正する。
ρlとγlは、l番目の層のリザーバーと入力変換行列の学習可能なスケーリング係数である。ESNMT-LSTMのLSTM状態遷移式にも同様の修正が適用される。
---

Our models are trained with back-propagation and
cross-entropy loss as usual.1 Note that since recurrent layer weights are fixed and their gradients are not calculated, the challenging gradient explosion/vanishing problem (Pascanu et al., 2013) commonly observed in training RNNs can be significantly alleviated. Therefore we expect no significant difference in quality between ESNMT
and ESNMT-LSTM, since the LSTM architecture,
which was originally designed to tackle the gradient instability problem, will not be superior in this
case. This is verified in our experimental results
(Sec. 4.2).
---
我々のモデルは、通常通りバックプロパゲーションとクロスエントロピー損失でトレーニングされる。リカレント層の重みは固定されており、その勾配は計算されないため、RNNのトレーニングで一般的に観察される勾配爆発/消失問題（Pascanu et al., 2013）は大幅に緩和されると期待される。したがって、ESNMTとESNMT-LSTMの間で品質に大きな違いはないと予想される。これは、元々勾配の不安定性問題に対処するために設計されたLSTMアーキテクチャが、この場合には優位ではないためである。このことは、実験結果（Sec. 4.2）で確認されている。
---

Since randomized components of ESNMT can be
deterministically generated simply from one fixed
random seed, to store the model offline we only
need to save this single seed together with remaining trainable model parameters. For example, in
an ESNMT-LSTM model with 6-layer encoder and
decoder of dimension 512 and vocabulary size 32K,
around 52% of the parameters from the recurrent
layers can be recovered from a single random seed.
---
ESNMTのランダム化されたコンポーネントは、1つの固定されたランダムシードから決定論的に生成できるため、モデルをオフラインで保存するには、この単一のシードと残りの学習可能なモデルパラメータを保存するだけでよい。例えば、6層のエンコーダーとデコーダーを持つESNMT-LSTMモデルで、次元512、語彙サイズ32Kの場合、リカレント層からのパラメータの約52%は1つのランダムシードから再現できる。
---

We train and evaluate our models on WMT’14
English→French, English→German and WMT’16
English→Romanian datasets. Sentences are processed into sequences of sub-word units using BPE
(Sennrich et al., 2016). We use a shared vocabulary
of 32K sub-word units for both source and target
languages.
Our baselines are fully trainable RNMT+ with
LSTM cells. For the proposed ESNMT models,
all reservoir and input transformation matrices are
generated randomly from a uniform distribution between -1 and 1. The reservoirs are then randomly
pruned so that Wres and Win reach 20-25% sparsity2
, and normalizeWres so that its spectral radius
equals to 1. Note the effective spectral radius and
input scaling are determined by the learnable scaling factors as shown in Eq. 2, which are initialized
to 1 and 10 respectively for all layers. For all models the number of encoder and decoder layers are
equally set to 6, and model dimension to 512 or
2048. We also adopt similar training recipes as
used by the RNMT+ (Chen et al., 2018), including
dropout, label smoothing and weight decay for all
our models.
---
我々は、WMT’14の英語→フランス語、英語→ドイツ語、およびWMT’16の英語→ルーマニア語データセットでモデルをトレーニングおよび評価する。文はBPE（Sennrich et al., 2016）を使用してサブワードユニットのシーケンスに処理される。ソース言語とターゲット言語の両方に対して、32Kのサブワードユニットの共有語彙を使用する。
我々のベースラインは、LSTMセルを持つ完全に学習可能なRNMT+である。提案するESNMTモデルでは、すべてのリザーバーと入力変換行列は、-1から1の一様分布からランダムに生成される。その後、リザーバーはランダムにプルーニングされ、WresとWinは20-25%のスパース性に達し、Wresを正規化してそのスペクトル半径が1になるようにする。効果的なスペクトル半径と入力スケーリングは、Eq. 2に示される学習可能なスケーリング係数によって決定され、すべての層でそれぞれ1と10に初期化される。すべてのモデルで、エンコーダーとデコーダーの層数は6に等しく、モデルの次元は512または2048に設定される。また、RNMT+（Chen et al., 2018）で使用されるのと同様のトレーニングレシピを採用し、ドロップアウト、ラベルスムージング、およびすべらのモデルに対する重み減衰を含む。
---

The results show that ESNMT can reach 70-80%
of the BLEU scores yielded by fully trainable baselines across all settings. Moreover, using LSTM
cells yields more or less the same performance as
a simple RNN cell. This verifies our hypothesis
in Sec. 3.1 that an LSTM cell is not particularly
advantageous compared to a simple RNN cell in
the ESN setting.
---
結果は、ESNMTがすべての設定で完全に学習可能なベースラインによって得られるBLEUスコアの70-80%に達することを示している。さらに、LSTMセルを使用すると、単純なRNNセルとほぼ同じ性能が得られる。これは、ESN設定において、LSTMセルが単純なRNNセルと比較して特に有利ではないという仮説を検証している。
---

As mentioned in Sec 3.1, in addition to randomizing both the encoder and decoder,
we explore other strategies of applying randomization, and conduct an ablation test as follows: We
start by randomizing and freezing everything in the
ESNMT-LSTM model (dimension 512) except the
softmax layer, then gradually release attention, encoder and/or decoder so that they become trainable.
The results for En→Fr are shown in Table 2.
---
前述のように、エンコーダーとデコーダーの両方をランダム化するだけでなく、ランダム化を適用する他の戦略も探り、以下のようなアブレーションテストを実施する。ESNMT-LSTMモデル（次元512）でソフトマックス層を除くすべてをランダム化して固定し、アテンション、エンコーダー、および/またはデコーダーを徐々に解放して学習可能にする。En→Frの結果を表2に示す。
---

From the table we have the following interesting
findings:
1. By randomizing only the entire decoder, the
BLEU score (37.98) drops only by 1.17 from the
baseline (39.15).
2. Randomizing the encoder incurs more BLEU
loss (35.21) than decoder. This shows that training the encoder properly is more critical to seq2seq
tasks.
3. Embedding layer deserves the most training. It
lifts the BLEU given by an almost purely randomized model (4.44) immediately to 26.63. It is also
interesting to note that a model with only the embedding and softmax layers trainable is already
able to reach this BLEU score.
---
表から、以下の興味深い発見が得られる。
1. デコーダー全体のみをランダム化することで、BLEUスコア（37.98）はベースライン（39.15）からわずか1.17ポイント低下する。
2. エンコーダーをランダム化すると、デコーダーよりもBLEU損失が大きくなる（35.21）。これは、エンコーダーを適切に学習させることがseq2seqタスクにとってより重要であることを示している。
3. 埋め込み層は最も学習に値する。ほぼ純粋にランダム化されたモデル（4.44）によって与えられるBLEUをすぐに26.63に引き上げる。また、埋め込み層とソフトマックス層のみが学習可能なモデルでも、このBLEUスコアに到達できることも興味深い。
---

To find out why ESNMT works, we examine learned spectral radii ρ
l
for each layer, which are are critical in characterizing the dynamics of ESNMT. In Fig. 1 we show the
learning curves of ρ
l
for all layers in the forward
encoder ESN. The figure shows a clear trend that
the radius increases almost monotonically from
bottom to top layer (0.55 to 1.8). This indicates
that lower layers retain short memories and focus
more on word-level representations, while upper
layers keep longer memories and account for better
sentence-level semantics which requires capturing
long-term dependencies between inputs. Similar
phenomena are observed for the backward encoder
ESN and decoder.
---
ESNMTがなぜ機能するのかを調べるために、各層の学習されたスペクトル半径ρlを調べる。これはESNMTの動力学を特徴付ける上で重要である。図1に、フォワードエンコーダーESNのすべての層に対するρlの学習曲線を示す。この図は、半径が下層から上層へとほぼ単調に増加する明確な傾向を示している（0.55から1.8へ）。これは、下層が短期記憶を保持し、単語レベルの表現に焦点を当てる一方で、上層は長期記憶を保持し、入力間の長期依存関係を捉える必要があるより良い文レベルのセマンティクスを考慮していることを示している。後方エンコーダーESNとデコーダーでも同様の現象が観察される。
---

To further investigate how spectral radius determines translation quality, we study BLEU scores
on EnFr testset as a function of sentence length, using models in which radii ρ
l
are fixed for all layers
and set to 0.1, 0.9 or 2.0. The results are shown
in Fig. 2, from which we see that when the radius
is small (0.1), the model favors shorter sentences
which requires less memory, increasing the radius
to 2.0 equips the model with non-fading memory,
in which remote inputs outweigh recent inputs, resulting in worse quality on short sentences. Radius
0.9 maintains a good balance between short and
long memories, yielding the best quality. Nevertheless the overall quality for all settings is worse than models whose radii are learned (Table 1).
---
スペクトル半径が翻訳品質をどのように決定するかをさらに調査するために、すべての層で半径ρlが固定され、0.1、0.9、または2.0に設定されたモデルを使用して、EnFrテストセットの文の長さに対するBLEUスコアを調べる。結果を図2に示す。ここから、半径が小さい（0.1）場合、モデルは短い文を好み、これはメモリを必要としないことがわかる。半径を2.0に増やすと、モデルは消えないメモリを持ち、遠くの入力が最近の入力を上回り、短い文での品質が悪化する。半径0.9は短期記憶と長期記憶のバランスをうまく保ち、最良の品質をもたらす。それにもかかわらず、すべての設定での全体的な品質は、半径が学習されるモデルよりも劣る（表1）。
---

We proposed Echo State NMT models whose encoder and decoder are composed of randomized
and fixed ESN layers. Even without training these
major components, the model can already reach 70-
80% performance yielded by fully trainable baselines. These surprising findings encourage us to
rethink about the nature of encoding and decoding
in NMT, and design potentially more economic
model architectures and training procedures.
ESNMT is based on the recurrent network architecture. One interesting research problem for
future exploration is how to apply randomized algorithms to non-recurrent architectures like Transformer (Vaswani et al., 2017). This is potentially
possible, as exemplified by randomized feedforward networks like Extreme Learning Machine
(Huang et al., 2006).
---
我々は、エンコーダーとデコーダーがランダム化され固定されたESN層で構成されるエコー状態NMTモデルを提案した。これらの主要なコンポーネントをトレーニングせずとも、モデルは完全に学習可能なベースラインによって得られる性能の70-80%に達することができる。この驚くべき発見は、NMTにおけるエンコードとデコードの本質について再考し、より経済的なモデルアーキテクチャとトレーニング手順を設計することを促す。
ESNMTはリカレントネットワークアーキテクチャに基づいている。将来の探求のための興味深い研究問題は、
ランダム化アルゴリズムをTransformer（Vaswani et al., 2017）のような非リカレントアーキテクチャに適用する方法である。これは、Extreme Learning Machine（Huang et al., 2006）のようなランダム化されたフィードフォワードネットワークの例からも可能である。
---

Unlike RNMT+ which is fully trainable, we
simply replace all recurrent layers in the encoder
and decoder with echo state layers as shown in
Eq. 1, and call this model ESNMT.
---
ESNMTは、RNMT+のように完全に学習可能ではなく、エンコーダーとデコーダーのすべてのリカレント層をエコー状態層に置き換え（Eq. 1参照）、このモデルをESNMTと呼ぶ。
---

Inspired by the simple yet effective construction
of ESN, we are interested in extending ESN to
challenging sequence-to-sequence prediction tasks,
especially NMT. We propose an ESN-based NMT
model whose architecture follows RNMT+ (Chen
et al., 2018), the state-of-the-art RNN-based NMT model.
---
ESNのシンプルでありながら効果的な構造に触発され、我々はESNを挑戦的なシーケンスツーシーケンス予測タスク、特にNMTに拡張することに興味を持つ。RNMT+（Chen et al., 2018）に従ったアーキテクチャを持つESNベースのNMTモデルを提案する。
---