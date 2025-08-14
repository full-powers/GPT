

T
RANSFORMERS [1] are a general purpose model with
a wide range of applications, including Machine Translation [1], Natural Language Processing (NLP) [2], Genome
Sequencing [3] and Computer Vision [4]. The core component
of the Transformer is the Scaled Dot-product Attention (SDA),
which receives three matrices Q, K and V as inputs, where
each row of which corresponds to a token introduced to the
SDA, computed by the linear transformations Q = XQWQ,
K = XKWK and V = XV WV . In the case of self-attention,
all three input matrices are identical, i.e., XQ = XK = XV ,
while when SDA is used to implement cross-attention of two
inputs, XK = XV corresponds to the first and XQ to the
second input data to be fused by the SDA. Q, K and V are
then used to perform the following transformation:

--

トランスフォーマー[1]は、機械翻訳[1]、自然言語処理(NLP)[2]、ゲノム配列決定[3]、コンピュータビジョン[4]など、幅広い用途に使用される汎用モデルです。トランスフォーマーのコアコンポーネントはスケールドドット積アテンション(SDA)であり、Q、K、およびVの3つの行列を入力として受け取ります。これらの行列の各行は、線形変換Q = XQWQ、K = XKWK、およびV = XV WVによってSDAに導入されたトークンに対応します。自己注意の場合、3つの入力行列はすべて同一であり、すなわちXQ = XK = XVですが、SDAが2つの入力のクロスアテンションを実装するために使用される場合、XK = XVは最初の入力データに対応し、XQはSDAによって融合される2番目の入力データに対応します。次に、Q、K、およびVを使用して次の変換を実行します。

-

where s(·) is the softmax function applied row-wise on its
input. The above computations on the input matrices XQ, XK
and XV are commonly performed multiple times in parallel,
leading to multiple SDA-based data transformations, often
referred to Multi-Head Attention. Thanks to this attention mechanism, the model can establish relations between the
different tokens in the input sequence, which is the key aspect
giving Transformers a great capacity to solve a wide variety of
tasks. Even though their original formulation targeted language
data, it has been shown that other types of data like images can
be processed by Transformer models through tokenization, as
done in the Vision Transformers [5].
Each of the three input matrices in Eq. (1) has a dimensionality of n × d, where n is the number of tokens and d the
number of features, leading the computational and memory
costs of the SDA module to have a quadratic form O(n
2d). For
large n, this attention mechanism can be too expensive, both
computationally and memory-wise, for certain tasks where
model inference needs to be performed with limited resources,
such as tasks coming from applications in Robotic Perception
and Control [6], [7], Forecasting [8], and Computer Vision [9],
[10].

--

ここで、s(·)はその入力に行単位で適用されるソフトマックス関数です。上記の計算は、通常、並列で複数回実行され、複数のSDAベースのデータ変換が行われます。これはしばしばマルチヘッドアテンションと呼ばれます。このアテンションメカニズムのおかげで、モデルは入力シーケンス内の異なるトークン間の関係を確立でき、さまざまなタスクを解決するための大きな能力を持つトランスフォーマーの重要な側面となっています。元々は言語データを対象としていたにもかかわらず、Vision Transformers[5]で行われているように、トークン化を通じて画像などの他のタイプのデータもトランスフォーマーモデルによって処理できることが示されています。
各行列の3つの入力行列は、n × dの次元を持ち、ここでnはトークンの数、dは特徴の数であり、SDAモジュールの計算コストとメモリコストはO(n
2d)の二次形式を持ちます。nが大きい場合、このアテンションメカニズムは、ロボティック知覚と制御[6]、予測[8]、コンピュータビジョン[9]、 [10]などのアプリケーションからのタスクなど、限られたリソースでモデル推論を実行する必要がある特定のタスクに対して、計算的にもメモリ的にも高価すぎる場合があります。

-

Different approaches have been proposed to improve the
efficiency of Transformer-based models. The simplest way is
to reduce the size of the model and, in particular, the number
of Transformer layers and attention heads [11]. However,
due to the much lower number of learnable parameters, the
resulting model may have insufficient capacity to perform well
on complex tasks. This creates the need for structural model
modifications that reach a better compromise between learning
capacity and resource requirements. One approach for reducing the overall cost of the attention product is to reduce the
number of features per token [12], [13]. Another approach is to
limit the attention window of each token, as done by the Sparse
Transformer [14] which reduces the attention window of each
token to √
n adjacent tokens, leading to a computational cost
of O(n
√
n). The Longformer [15] generalizes this concept by
using a small fixed window around each token to capture local
patterns and a dilated sliding window (where not all tokens are
used) to perform the attention efficiently. The Performer [16]
approximates the softmax attention with a Gaussian kernel
by selecting a set of orthogonal random features to represent
the entirety of the attention window. A different approach is
to use low-rank matrix approximation schemes [17], [18] for
reducing both the number of computations and the size of the
matrices involved in the SDA, as done by the Nystromformer ¨
[19] which approximates the matrix multiplication followed
by softmax nonlinearity in SDA. Efficient designs of Transformer model architectures also include the aggregation of
neighboring tokens [20], the addition of convolutional layers to
reduce the input dimensionality [21], and architectures which
tackle challenges of processing videos [22] such as using
different restricted attentions [23], [24], or aggregating features
to reduce the number of tokens in the sequence [25], [26].

--

さまざまなアプローチが、トランスフォーマーベースのモデルの効率を改善するために提案されています。最も簡単な方法は、モデルのサイズ、特にトランスフォーマー層とアテンションヘッドの数を減らすことです[11]。ただし、学習可能なパラメータの数がはるかに少ないため、結果として得られるモデルは、複雑なタスクでうまく機能するための能力が不十分になる可能性があります。これにより、学習能力とリソース要件との間でより良い妥協点に到達するための構造的なモデル変更の必要性が生じます。アテンション積の全体的なコストを削減する1つのアプローチは、トークンごとの特徴量の数を減らすことです[12][13]。別のアプローチは、各トークンのアテンションウィンドウを制限することであり、Sparse Transformer[14]では、各トークンのアテンションウィンドウを√n隣接トークンに減らし、計算コストをO(n√n)にしています。Longformer[15]は、この概念を一般化し、各トークンの周りに小さな固定ウィンドウを使用してローカルパターンをキャプチャし、拡張スライディングウィンドウ(すべてのトークンが使用されるわけではない)を使用してアテンションを効率的に実行します。Performers[16]は、正規直交ランダム特徴のセットを選択してアテンションウィンドウ全体を表現することにより、ガウスカーネルでソフトマックスアテンションを近似します。別のアプローチは、SDAで関与する行列の計算数とサイズの両方を削減するために低ランク行列近似スキーム[17][18]を使用することであり、Nystromformer ¨[19]は、SDAでソフトマックス非線形性の後に行列乗算を近似します。トランスフォーマーモデルアーキテクチャの効率的な設計には、隣接トークンの集約[20]、入力次元を削減するための畳み込み層の追加[21]、およびビデオ処理の課題に取り組むアーキテクチャ[22]が含まれています。 [23][24]やシーケンス内のトークン数を減らすために特徴量を集約する[25][26]などです。

-

We propose a new formulation of the Continual Transformer, which further improves its memory and computational cost requirements for Continual Inference settings.
We incorporate the Nystrom-based approximation of the ¨
matrix multiplication followed by the softmax nonlinearity in SDA, to further lower the Continual SDA’s memory
footprint and the number of computations compared to
its original formulation. To do this, we derive the model
updates of the Nystrom-based SDA in a continual manner.

--

新しい継続的トランスフォーマーの定式化を提案します。これは、継続的推論設定のメモリと計算コスト要件をさらに改善します。
Nystromベースの近似を組み込みます。SDAのソフトマックス非線形性に続く行列乗算は、元の定式化と比較して、継続的なSDAのメモリフットプリントと計算数をさらに低下させます。これを行うために、継続的な方法でNystromベースのSDAのモデル更新を導出します。

-

We propose two different ways to determine the landmarks used for processing continual stream data in the
SDA approximation, and make the corresponding module
modifications.

--

2つの異なる方法を提案します。SDA近似で継続的なストリームデータを処理するために使用されるランドマークを決定し、対応するモジュール変更を行います。

-

While efficient model architectures have been proposed
that can lead to fast inference for static tasks, such as the
classification of images, videos or audio sequences, Continual
Inference tasks remain challenging for Deep Learning models
(even for those models that can operate in real time on the
corresponding static task). Continual Inference [27] can be
defined as the process of providing a result for each input of
a continual data stream, for instance performing event/action
classification based on the visual frames captured by a camera
operating continually for a long period of time. An inherent
requirement for solutions targeting Continual Inference tasks
is that they need to be able to process the incoming stream of
data with low latency and low resource consumption.

--

効率的なモデルアーキテクチャが提案されており、静的タスクの高速推論を実現できます。たとえば、画像、ビデオ、またはオーディオシーケンスの分類などです。継続的な推論タスクは、ディープラーニングモデルにとって依然として課題です(対応する静的タスクでリアルタイムで動作できるモデルでも)。継続的な推論[27]は、継続的なデータストリームの各入力に対して結果を提供するプロセスとして定義できます。たとえば、長期間にわたって継続的に動作するカメラによってキャプチャされた視覚フレームに基づいてイベント/アクションクラシフィケーションを実行します。継続的な推論タスクを対象としたソリューションの固有の要件は、低遅延と低リソース消費でデータストリームを処理できる必要があることです。

-

The proposed method targets (approximate) Transformer inference with lower computational and memory requirements.
This is done by adapting the Nystrom-based approximation of ¨
the matrix multiplication followed by softmax nonlinearity so
that it can be performed in a Continual Inference setting [27].
In the following, we provide an overview of the Continual
Transformer and the Nystrom-based formulation of SDA, ¨
which form the basis of our work.

--

提案された方法は、計算とメモリ要件が低い(近似)トランスフォーマー推論を対象としています。
これは、ソフトマックス非線形性に続く行列乗算のNystromベースの近似を適応させることによって行われます。継続的な推論設定[27]で実行できるようにします。以下では、継続的トランスフォーマーとSDAのNystromベースの定式化の概要を説明します。¨
これらは、私たちの作業の基礎を形成します。

-

The Continual Transformer [28] adapts the formulation of
the SDA in Eq. (1) for Continual Inference settings, leading
to the Continual Retroactive Attention formulation, which
reuses computations performed at prior inference steps. The
Continual Retroactive Attention is defined as follows:

--

継続的なトランスフォーマー[28]は、継続的な推論設定のSDAの定式化を適応させ、以前の推論ステップで実行された計算を再利用する継続的な遡及アテンションの定式化を導き出します。継続的な遡及アテンションは次のように定義されます。

-


In the above, qnew, knew and vnew are the newest tokens
corresponding to the new query, key and value, respectively, ⊙
denotes a column-aligned element-wise multiplication operation, 1n is a row-vector of n ones,

--

上記では、qnew、knew、およびvnewは、それぞれ新しいクエリ、キー、および値に対応する最新のトークンであり、⊙は列揃えの要素ごとの乗算操作を示し、1nはn個の1の行ベクトルです。



where the AV matrix is the result of the multiplication
between the A and V matrices, ϕ(Aˆ) and AVˆ are the matrices
ϕ(A) and AV from the previous update without their first row,
respectively, kold is the token that shifts out of the attention
window when knew enters, and Qmem are the n−1 rows of the
query matrix that are still part of the attention window.
In the Continual Retroactive Attention, new tokens update
the attention values of all the previous tokens within the
attention window. This allows the Continual Transformers to
achieve linear computational and memory cost of O(nd). If
only the newest token inference is needed, the Single Output
Attention can be used to save some additional computations
and memory space.

--

ここで、AV行列はAとV行列の乗算の結果であり、ϕ(Aˆ)とAVˆはそれぞれ最初の行を除いた前の更新からの行列ϕ(A)とAVであり、koldはknewが入るときにアテンションウィンドウからシフトアウトするトークンであり、Qmemはアテンションウィンドウの一部であるクエリ行列のn−1行です。
継続的な遡及アテンションでは、新しいトークンがアテンションウィンドウ内のすべての以前のトークンのアテンション値を更新します。これにより、継続的なトランスフォーマーはO(nd)の線形計算コストとメモリコストを達成できます。最新のトークン推論のみが必要な場合は、シングル出力アテンションを使用して、追加の計算とメモリスペースを節約できます。

-

In this section, we describe in detail the proposed Continual
Transformer model employing a Nystrom-based formulation ¨
of SDA. Since the SDA approximation in Eq. (8) was
originally proposed for defining the Nystromformer model ¨
in [19], we refer to our proposed model as the Continual
Nystromformer. The Continual Nystr ¨ omformer adapts the ¨
computations needed for the Nystrom-based SDA in order to ¨
be performed in a Continual Inference setting [27]. We define
the Continual Nystrom-based SDA as: 

--

このセクションでは、SDAのNystromベースの定式化を採用した提案された継続的トランスフォーマーモデルについて詳しく説明します。SDA近似は、元々[19]でNystromformerモデルを定義するために提案されたため、提案されたモデルを継続的なNystromformerと呼びます。継続的なNystr ¨ omformerは、継続的な推論設定[27]で実行できるようにするために、NystromベースのSDAに必要な計算を適応させます。継続的なNystromベースのSDAを次のように定義します。

-

where the matrices Q and K are updated in a continual manner
when new tokens are received and Q˜ and K˜ are the landmark
matrices used for obtaining the approximation.

--

行列QとKは、新しいトークンが受信されるときに継続的な方法で更新され、Q˜とK˜は近似を取得するために使用されるランドマーク行列です。

-

As can be seen, in Eqs. (9)-(12), the landmark matrices Q˜
and K˜ are involved in the calculation of in all three matrices
B, Γ and ∆. The Nystromformer [19] calculates new land- ¨
marks for every inference step. However, this approach would
lead to computational redundancies in a Continual Inference
setting, as landmarks would need to be fully recomputed
after every inference step. To address this issue, we exploit
properties stemming from the fact that successive inference
steps involve processing of highly-overlapping sequence data.
We propose two ways for landmark selection, leading to model
updates described in Sections IV-A and IV-B. Considerations related to the implementation aspects of the proposed
model and its updates are discussed in Section IV-C. Table I
shows the asymptotic computational and memory costs of the
proposed method compared to the existing ones. A detailed
analysis of this can be found in A and B.

--

Eqs. (9)-(12)では、ランドマーク行列Q˜とK˜がすべての3つの行列B、Γ、および∆の計算に関与していることがわかります。Nystromformer[19]は、各推論ステップの新しいランドマークを計算します。ただし、このアプローチは、継続的な推論設定で計算の冗長性につながります。ランドマークは、各推論ステップの後に完全に再計算する必要があるためです。この問題に対処するために、連続的な推論ステップには、高度に重複したシーケンスデータの処理が含まれるという事実から生じる特性を利用します。ランドマーク選択の2つの方法を提案し、セクションIV-AおよびIV-Bで説明されているモデル更新につながります。提案されたモデルとその更新の実装に関連する考慮事項については、セクションIV-Cで説明します。表Iは、既存のものと比較して提案された方法の漸近的な計算コストとメモリコストを示しています。これについての詳細な分析はAおよびBで見つけることができます。

-

By observing the right-hand side of Eq. (1), one can make
connections of the SDA (specifically its first term) to dotproduct formulations appearing in kernel machines [32]. Thus,
when the objective is to define an approximate formulation of
SDA for large numbers of n, matrix approximation schemes
like the Nystrom approximation [17], [33] can be used. This ¨
idea was proposed in [19] to define the Nystromformer model, ¨
which approximates the softmax matrix of the SDA in the
corresponding attention AttNy(Q, K, V ) as follows:

--

Eq. (1)の右辺を観察することにより、SDA(特にその最初の項)とカーネルマシン[32]に現れるドット積定式化との接続を行うことができます。したがって、目的がnの大きな数のSDAの近似定式化を定義することである場合、Nystrom近似[17][33]などの行列近似スキームを使用できます。このアイデアは、対応するアテンションAttNy(Q, K, V )のSDAのソフトマックス行列を次のように近似するために、Nystromformerモデルを定義するために[19]で提案されました。

-

where Ω† is the Moore-Penrose pseudo-inverse of matrix Ω,
and Q˜ and K˜ are matrices formed by sets of m landmarks,
computed as the segment-means of the matrices Q and K,
respectively. When m ≪ n, it leads to a large reduction of
both costs compared to the standard SDA formulation in Eq.
(1), i.e., O(nd) computational and memory costs. The same
formulation has been used in [34] where, instead of defining
the landmarks as the segment-means, landmarks are chosen
from the sequence incrementally, in such a way that at each
step the most orthogonal vector to the already selected tokens
is chosen. In [35], attention is computed over the parts of
the image of a video where the most elements have changed,
triggering m updates over the image as the rest of the weights
are re-used.

--

ここで、Ω†は行列Ωのムーア-ペンローズ擬似逆行列であり、Q˜とK˜はそれぞれQとKの行列のセグメント平均として計算されたmランドマークのセットによって形成される行列です。m ≪ nの場合、Eq。 (1)の標準SDA定式化と比較して、計算コストとメモリコストの両方が大幅に削減されます。つまり、O(nd)の計算コストとメモリコストです。同じ定式化は[34]で使用されています。ランドマークをセグメント平均として定義する代わりに、ランドマークはシーケンスから徐々に選択され、各ステップで既に選択されたトークンに最も直交するベクトルが選択されるようにします。[35]では、ビデオの画像の部分に対してアテンションが計算されます。最も多くの要素が変更されており、残りの重みが再利用されるにつれて、画像全体でm更新をトリガーします。

-

Continual landmarks’ calculation (for n = 20 and m = 4). Until
enough new input tokens have been received, the landmarks remain fixed.
This includes the landmark calculated by some old tokens which have been
shifted out of the current attention window (top). When enough input tokens
have been received (bottom), a new landmark is computed using all the new
tokens, replacing the q˜old landmark. The same approach is applied to update K˜.

--

継続的なランドマークの計算(n = 20、m = 4)。十分な新しい入力トークンが受信されるまで、ランドマークは固定されたままです。これは、現在のアテンションウィンドウからシフトアウトしたいくつかの古いトークンによって計算されたランドマークを含みます(上)。十分な入力トークンが受信されると(下)、すべての新しいトークンを使用して新しいランドマークが計算され、q˜oldランドマークが置き換えられます。同じアプローチがK˜を更新するために適用されます。

-

Following the process of receiving new input tokens updating
the matrices Q and K in a continual manner, the landmark
matrices Q˜ and K˜ can be updated periodically, i.e., after a
pre-defined number of input tokens are received, as illustrated
in Figure 1. This means that the landmark matrices Q˜ and
K˜ are updated after receiving n/m input tokens, following
the segment-means process used in [19]. In practice, this will
cause the oldest landmarks q˜old and ˜kold (computed by using
tokens that have already been shifted out of the Q and K
matrices) to be shifted out of the landmark matrices Q˜ and K˜ ,
respectively. The newest landmarks q˜new and ˜knew, computed
as the mean of the most recent n/m tokens in the Q and K
matrices, respectively, will then be included in the landmark
matrices Q˜ and K˜ . As such, the update of the landmark
matrices takes the form:

--

新しい入力トークンを受信するプロセスに従って、行列QとKを継続的な方法で更新すると、ランドマーク行列Q˜とK˜は定期的に更新できます。つまり、図1に示すように、入力トークンの事前定義された数を受信した後です。これは、ランドマーク行列Q˜とK˜がn/m入力トークンを受信した後に更新されることを意味します。[19]で使用されるセグメント平均プロセスに従います。実際には、最も古いランドマークq˜oldとk˜old(すでにQとK行列からシフトアウトされたトークンを使用して計算されたもの)が、それぞれランドマーク行列Q˜とK˜からシフトアウトされます。最新のランドマークq˜newとk˜newは、それぞれQとK行列の最近のn/mトークンの平均として計算され、その後、ランドマーク行列Q˜とK˜に含まれます。このようにして、ランドマーク行列の更新は次の形式になります。

-

where qi and ki correspond to the i
th row of matrices Q and
K, respectively.
Updating the matrices B, Γ and ∆ based on continual
landmarks takes two forms, depending on whether the landmark matrices are have been updated with newly computed
landmarks or not, which are described in the following.

--

qiとkiは、それぞれ行列QとKのi行目に対応します。
行列B、Γ、および∆を継続的なランドマークに基づいて更新することは、ランドマーク行列が新しく計算されたランドマークで更新されたかどうかに応じて2つの形式を取り、次に説明します。

-


We also propose a process for determining appropriate landmarks during the training phase, which can then be used for
processing any received input, avoiding the need to perform
landmark updates during Continual Inference. This approach is
motivated by similar ideas used in approximate kernel-based
learning [36], [37], where landmarks in Nystrom-based ap- ¨
proximation of the kernel matrix are determined by clustering
the training data. However, this approach cannot be directly
applied in our case, as the data transformations performed by
all layers before each of the SDA blocks change at every
training update, leading to different feature spaces in which
the matrices XQ and XK are defined

--

トレーニングフェーズ中に適切なランドマークを決定するプロセスも提案します。これにより、継続的な推論中にランドマークの更新を実行する必要がなくなります。このアプローチは、カーネル行列のNystromベースの近似で使用される類似のアイデアに基づいています[36][37]。ここでは、トレーニングデータをクラスタリングすることによってランドマークが決定されます。ただし、このアプローチは、すべてのSDAブロックの前にあるすべてのレイヤーで実行されるデータ変換が各トレーニング更新で変更されるため、直接適用できません。これにより、行列XQとXKが定義される異なる特徴空間が得られます。

-


To address this issue, the training process is divided into two
phases. In the first phase, the model is trained in an end-to-end
manner using continually updated landmarks as described in
Section IV-A. The second phase is divided into two processing
steps. In the first step, the training data is introduced to the
model and the matrices Q and K are calculated for each input
data sample. The Q-tokens corresponding to all training data
are combined to create a dataset which is clustered into m
clusters by applying the m-Means method. The cluster centers
are then used to form the matrix Q˜. The same process is
applied to the K-tokens to form the matrix K˜ . If multiple
SDA heads are used, we compute the landmarks of each head
independently. In the second step, the model is fine-tuned in
an end-to-end manner using the now fixed, landmarks (i.e.,
the matrices Q˜ and K˜ are not updated). When the model is
formed by multiple SDA blocks, the two steps of phase two are
applied sequentially starting from the first block, and keeping
all landmarks of previous SDA blocks fixed in the fine-tuning
step. This leads to gradually determining all landmarks of the
model.

--

この問題に対処するために、トレーニングプロセスは2つのフェーズに分割されます。最初のフェーズでは、セクションIV-Aで説明したように、継続的に更新されたランドマークを使用して、モデルがエンドツーエンドの方法でトレーニングされます。2番目のフェーズは、2つの処理ステップに分かれています。最初のステップでは、トレーニングデータがモデルに導入され、各入力データサンプルについて行列QとKが計算されます。すべてのトレーニングデータに対応するQトークンを組み合わせて、m-Meansメソッドを適用してmクラスタにクラスタリングされたデータセットを作成します。次に、クラスタセンターを使用して行列Q˜を形成します。同じプロセスがKトークンにも適用され、行列K˜が形成されます。複数のSDAヘッドが使用される場合は、それぞれのヘッドのランドマークを独立して計算します。2番目のステップでは、今や固定されたランドマーク(つまり、行列Q˜とK˜は更新されません)を使用して、モデルがエンドツーエンドの方法で微調整されます。モデルが複数のSDAブロックで構成されている場合は、最初のブロックから始めて2つのステップを順次適用し、微調整ステップで前のSDAブロックのすべてのランドマークを固定します。これにより、モデルのすべてのランドマークが徐々に決定されます。
-


After training the model and determining all landmarks, the
SDA module used for Continual Retroactive Inference, i.e.,
AttFix
CoNyRe, has the form of Eq. (23), and the SDA module for
the Continual Single Output Inference, i.e., AttFix
CoNySi, has the
form of Eq. (26). The computational cost of this model is
identical to the Nystrom-based Continual Inference with non- ¨
updated landmarks for both the Retroactive and Single Output
version

--

モデルをトレーニングし、すべてのランドマークを決定した後、継続的な遡及推論に使用されるSDAモジュール、つまりAttFix
CoNyReはEqの形式を持ちます。 (23)、および継続的なシングル出力推論のSDAモジュール、つまりAttFix
CoNySiはEqの形式を持ちます。 (26)。このモデルの計算コストは、RetroactiveとSingle Outputの両方のバージョンで、更新されていないランドマークを使用したNystromベースの継続的な推論と同じです。

-


The Continual Nystromformers share some of the practical ¨
aspects of Continual Transformers [28], due to the properties
of the involved continual computation

--

継続的なNystromformersは、関与する継続的な計算の特性により、継続的なトランスフォーマー[28]のいくつかの実用的な側面を共有します。

-

 The Continual Nystromformers require a circular posi- ¨
tional encoding, as when new input tokens are processed
its positional encoding needs to be appropriately related
to the positional encodings of the rest of the sequence.
• The ability for Continual Nystromformers to reuse pre- ¨
vious computations is hampered when multiple stacked
SDA blocks are used. This is caused by the need to
recompute the entire sequence for all the earliest SDA
blocks, as the attention needs to be propagated accordingly. Thus, all SDA blocks except the last one must be
of a regular Nystromformer or any other non-Continual ¨
Transformer.
• For training, we use a modified version of the noncontinual model with the circular positional encoding
described above and the corresponding landmark selection scheme as described in Sections IV-A and IV-B,
depending on whether continual or fixed landmarks are
used, respectively. We follow this approach as the noncontinual training processes are faster when the entire
sequence is available from the beginning, and both continual and non-continual SDA variants produce identical
results.

--

継続的なNystromformersは、円形の位置エンコーディングを必要とします。新しい入力トークンが処理されるとき、その位置エンコーディングはシーケンスの残りの部分の位置エンコーディングに適切に関連する必要があります。
• 複数のスタックされたSDAブロックが使用される場合、継続的なNystromformersの以前の計算を再利用する能力は妨げられます。これは、注意が適切に伝播する必要があるため、すべての最初のSDAブロックのシーケンス全体を再計算する必要があるためです。したがって、最後のものを除くすべてのSDAブロックは、通常のNystromformerまたは他の非継続的なトランスフォーマーである必要があります。
• トレーニングには、上記で説明した円形の位置エンコーディングと、セクションIV-AおよびIV-Bで説明されている対応するランドマーク選択スキームを使用して、継続的または固定されたランドマークがそれぞれ使用されるかどうかに応じて、非継続的なモデルの修正バージョンを使用します。シーケンス全体が最初から利用可能な場合、非継続的なトレーニングプロセスはより高速であり、継続的および非継続的なSDAバリアントの両方が同一の結果を生成するため、このアプローチに従います。

-

For replacing the softmax with exponential operations
in both the Continual Transformers and our Continual
Nystromformers numerical stability issues need to be consid- ¨
ered, as the calculation of the exponential can be prone to
overflow and underflow. This is addressed in most softmax
implementations (e.g., [39]) by employing the so-called stable
softmax variant [40] s(x)i = exp(xi−C)/
Pn
j=1 exp(xj −C).
By setting C to the maximum value in x possible overflow
and underflow issues are addressed, as at least one value will
be higher than zero after the calculation of the exponential
operations. However, this approach cannot be applied in the
continual versions of SDA, as the maximum values of x in
the attention window can change at every inference step. This
would cause constant updates in most matrices, increasing the
cost very significantly. In our experiments described in the
next section, this issue has not been observed. In case such a
stability issue is observed, using a dropout layer or other type
of normalization layer before the SDA module can address it.

--

ソフトマックスを継続的なトランスフォーマーと継続的なNystromformersの両方で指数演算に置き換える場合、指数の計算はオーバーフローやアンダーフローを引き起こす可能性があるため、数値的安定性の問題を考慮する必要があります。これは、いわゆる安定したソフトマックスバリアント[40]s(x)i = exp(xi−C)/Pn
j=1 exp(xj −C)を使用することによって、ほとんどのソフトマックス実装(e.g., [39])で対処されています。
Cをxの最大値に設定すると、指数演算の計算後に少なくとも1つの値がゼロより大きくなるため、オーバーフローとアンダーフローの問題が解決されます。ただし、このアプローチは、SDAの継続的なバージョンには適用できません。なぜなら、アテンションウィンドウ内のxの最大値は、各推論ステップで変更できるからです。これにより、ほとんどの行列で定数更新が発生し、コストが大幅に増加します。次のセクションで説明する実験では、この問題は観察されていません。このような安定性の問題が観察された場合、SDAモジュールの前にドロップアウトレイヤーまたは他のタイプの正規化レイヤーを使用すると、それを解決できます。
-

Fig.2
Continual Inference with updated landmarks for a sequence of n = 4 tokens, each having d = 5 dimensions, and using m = 3 landmarks. The red
elements represent the tokens that just exited the inference window (old), and the green elements represent the newly received tokens (new). The blue elements
are those exclusively used by the Retroactive Attention formulation. The elements with a red-black-green color symbolize tokens that need to be updated by
removing the influence of the oldest tokens and adding the influence of the newest tokens. This results in updates to all three B, Γ and ∆ matrices.

--

図2
継続的な推論は、n = 4トークンのシーケンスで、各トークンがd = 5次元を持ち、m = 3ランドマークを使用します。赤い要素は、推論ウィンドウからちょうど退出したトークン(古い)を表し、緑の要素は新しく受信したトークン(新しい)を表します。青い要素は、Retroactive Attention定式化によって独占的に使用されるものです。赤-黒-緑の色の要素は、最も古いトークンの影響を取り除き、新しいトークンの影響を追加することによって更新する必要があるトークンを象徴しています。これにより、B、Γ、および∆行列のすべてに更新が行われます。
-

Fig. 3. Continual Inference with non-updated landmarks. As landmarks remain unchanged, we can update most of the previous matrices to save computations.
The details on color use and dimensionality are identical to those in Figure 2.

--

図3。更新されていないランドマークによる継続的な推論。ランドマークが変更されないため、計算を節約するために以前の行列のほとんどを更新できます。色の使用と次元の詳細は、図2のものと同じです。
-

Illustration of a Nystrom approximation of softmax ma- ¨
trix in self-attention. The left image shows the true softmax matrix
used in self-attention and the right images show its Nystrom ap- ¨
proximation. Our approximation is computed via multiplication of
three matrices.

--

ソフトマックス行列のNystrom近似の説明 ¨
自己注意。左側の画像は、自己注意で使用される真のソフトマックス行列を示し、右側の画像はそのNystrom近似を示しています。 ¨
私たちの近似は、3つの行列の乗算を介して計算されます。
-

Note that we arrive at (13) via an out-of-sample approximation similar to (4). The difference is that in (13), the landmarks are selected before the softmax operation to generate the out-of-sample approximation. This is a compromise
but avoids the need to compute the full softmax matrix S
for a Nystrom approximation. Fig. 2 illustrates the proposed ¨
Nystrom approximation and Alg. 1 summarizes our method. ¨
We now describe (a) the calculation of the Moore-Penrose
inverse and (b) the selection of landmarks

--

注意してください。 (13)に到達するのは、(4)と同様のサンプル外近似を介して到達します。違いは、(13)では、ランドマークがサンプル外近似を生成するためにソフトマックス操作の前に選択されることです。これは妥協ですが、Nystrom近似の完全なソフトマックス行列Sを計算する必要がなくなります。図2は、提案されたNystrom近似を示し、アルゴリズム1は私たちの方法を要約しています。 ¨

-

Let us assume that a matrix Ω ∈ R
n×d
formed by n sequence
tokens is updated in a continual manner, i.e., its top-most
row corresponds to the oldest sequence token and its bottommost row corresponds to the newest sequence token. When an
update takes place, we define two tokens, i.e., ωold, which
corresponds to the oldest token included in Ω before the
update, and ωnew, which is the new token to be included by
the update. Then:
• Ωmem is formed by the n − 1 tokens already included in
Ω = 
ωold
Ωmem
which shift positions in the sequence such
that after the update we have Ω = 
Ωmem
ωnew 
.
• When an update takes place and Ωmem needs to be
updated to incorporate the influence of ωnew, we define
Ωˆ as Ωmem before the update.
The notation above can also be used for vectors (represented
with lowercase letters) where the corresponding new and old
elements will correspond to single values.

--

行列Ω ∈ R
n×d
が継続的な方法で形成されていると仮定しましょう。つまり、その最上部の行は、更新前のΩに含まれる最も古いシーケンストークンに対応し、最下部の行は最新のシーケンストークンに対応します。更新が行われると、ωoldを定義します。これは、更新前にΩに含まれる最も古いトークンに対応し、ωnewは更新によって含まれる新しいトークンです。次に:
• Ωmemは、Ω = 
ωold
Ωmem 
によってすでにΩに含まれているn − 1トークンによって形成されます。更新後、Ω = 
Ωmem
ωnew  
になります。
• 更新が行われ、ωnewの影響を組み込むためにΩmemを更新する必要がある場合、更新前のΩmemとしてΩˆを定義します。
この上記の表記法は、対応する新しい要素と古い要素が単一の値に対応するベクトル(小文字で表される)にも使用できます。
-

This
is the case where the newly received input tokens qnew, knew
and vnew do not lead to the calculation of a new set of
landmarks, thus the matrices Q˜ and K˜ remain identical to
those used in the previous inference step. An illustration of
the process followed in this case can be seen in Figure 3.
We define the following formulation, where updates involve
only the new input tokens qnew, knew and vnew:

--

これは、新しく受信した入力トークンqnew、knew、およびvnewが新しいランドマークのセットの計算につながらない場合です。したがって、行列Q˜とK˜は、前の推論ステップで使用されるものと同じままです。この場合に従うプロセスの説明は、図3に示されています。次の定式化を定義します。ここで、更新には新しい入力トークンqnew、knew、およびvnewのみが含まれます。

-

where (Bϕ(Γϕ)†)mem is the matrix corresponding to the n−1 most recent tokens of the matrix (Bϕ(Γϕ)†) from the previous iteration. The vector ϕ(∆)−1
and matrix ∆V require updates
to all of their elements, by removing the influence of the oldest
tokens and adding the influence of the newest tokens, which is done as follows:]

--
(Bϕ(Γϕ)†)memは、前の反復からの行列(Bϕ(Γϕ)†)のn−1の最も最近のトークンに対応する行列です。ベクトルϕ(∆)−1
および行列∆Vは、最も古いトークンの影響を取り除き、新しいトークンの影響を追加することによって、すべての要素を更新する必要があります。これは次のように行われます。
-

where ϕ(∆)−1
prev and (∆V )prev correspond to the matrices
obtained in the previous inference step.
Similarly to the continual landmark version, a Single Output
simplified version can be formulated, leading to:

--

ϕ(∆)−1
prevと(∆V )prevは、前の推論ステップで得られた行列に対応します。
同様に、継続的なランドマークバージョンと同様に、シングル出力の簡略化されたバージョンを定式化でき、次のようになります。

-

There exist multiple ways to count and parametrize the actual
computational cost of running a model for inference. A metric
that is extensively used is the number of floating operations (FLOPs) which corresponds to the number of element
computations required to perform inference. To study the
computational efficiency of the proposed SDA formulation
in comparison with other related formulations for different
lengths of input sequences n, we provide the number of
FLOPs for different sequence lengths when using a number
of dimensions d = 200 and m = 8 landmarks in Figure 4. As
the proposed method affects exclusively the SDA, the figure
illustrates the number of average FLOPs required for a single
prediction during sequential processing corresponding to the
SDA modules of the competing methods.

--

複数の方法があり、推論のためにモデルを実行する実際の計算コストをカウントしてパラメータ化します。広く使用されているメトリックは、推論を実行するために必要な要素計算の数に対応する浮動小数点演算(FLOP)です。異なる入力シーケンスnの長さに対して、他の関連する定式化と比較して提案されたSDA定式化の計算効率を研究するために、次の図4に、d = 200とm = 8ランドマークを使用した場合の異なるシーケンス長のFLOPの数を示します。提案された方法はSDAにのみ影響するため、図は競合する方法のSDAモジュールに対応する逐次処理中に単一の予測に必要な平均FLOPの数を示しています。

-

Since the proposed approach of determining the landmarks
in the training phase and fixing them for performing inference
for any input sample in the test phase can be used also
by the Nystromformer model, we also created a variant of ¨
the Nystromformer using fixed landmarks and illustrate its ¨
computational cost as AttFix
Ny in Figure 4. For the Continual
Inference models using one SDA block their single output
versions are used, while for those using two SDA blocks an
SDA with retroactive inference is followed by a single output
SDA.

--

提案されたランドマークを決定するアプローチは、トレーニングフェーズでランドマークを決定し、テストフェーズで任意の入力サンプルの推論を実行するためにそれらを固定することは、Nystromformerモデルでも使用できるため、固定されたランドマークを使用したNystromformerのバリアントも作成しました。 ¨
その計算コストをAttFixとして図4に示します。継続的な推論モデルは、1つのSDAブロックを使用している場合は、シングル出力バージョンが使用されますが、2つのSDAブロックを使用している場合は、レトロアクティブ推論を伴うSDAの後にシングル出力SDAがあります。
-

When measuring the memory overhead of a model, we need
to consider not only the matrices we need to store between
iterations, but also the necessary matrices to perform the
computations leading to the output. With this in mind, we
provided memory costs of the different SDA formulations
based on the dimensions of the input, the sequence length
and the number of landmarks in Section IV. We used these
to measure the peak memory overhead of the SDA module
with d = 200 and m = 8 for a varying sequence length n, as
shown in Figure 4.

--

モデルのメモリオーバーヘッドを測定する場合、反復間で保存する必要のある行列だけでなく、出力につながる計算を実行するために必要な行列も考慮する必要があります。このことを考慮して、セクションIVで、入力の次元、シーケンスの長さ、およびランドマークの数に基づいて、さまざまなSDA定式化のメモリコストを提供しました。これらを使用して、図4に示すように、さまざまなシーケンス長nについてd = 200およびm = 8のSDAモジュールのピークメモリオーバーヘッドを測定しました。
-

The first aspect to notice is that the original SDA formulation Att leads to the highest computational cost, which has a
quadratic asymptotic form. The SDA formulations AttNy and
AttCo used by the Nystromformer and Continual Transformer ¨
models have a similar asymptotic form for their computational
cost as the original SDA formulation, but with much lower
computations compared to the original SDA formulation,
while the Nystromformer has a higher number of computations ¨
than the Continual Transformer.

--

最初に注意すべき点は、元のSDA定式化Attが最高の計算コストにつながることであり、これは二次的な漸近的な形式を持っています。NystromformerとContinual Transformer ¨モデルによって使用されるSDA定式化AttNyとAttCoは、元のSDA定式化と同様の漸近的な形式を持っていますが、元のSDA定式化と比較してはるかに低い計算であり、NystromformerはContinual Transformerよりも計算数が多くなります。

-

For the proposed SDA formulations AttCont
CoNy and AttFix
CoNy
we need to distinguish two cases, i.e., when one or two
SDA blocks are used. When two SDA blocks are used, the
computational cost of the proposed SDA formulations has a
very similar form to that of the SDA formulation AttCo used
in the Continual Transformer, while being lower in absolute
numbers. The difference between these computational costs
depends on the selected number of landmarks m and, as
the number of sequence tokens n increases, the difference
in the computational costs between the two types of SDA
formulations increases

--

提案されたSDA定式化AttCont
CoNyとAttFix CoNyは、1つまたは2つのSDAブロックが使用される場合に区別する必要があります。
2つのSDAブロックが使用される場合、提案されたSDA定式化の計算コストは、継続的なトランスフォーマーで使用されるSDA定式化AttCoと非常に似た形式を持っていますが、絶対数では低くなります。これらの計算コストの違いは、選択したランドマークの数mに依存し、シーケンストークンnの数が増えるにつれて、2種類のSDA定式化間の計算コストの違いが増加します。

-

In this paper, we introduced a new formulation of the Scaled
Dot-product Attention based on the Nystrom approximation ¨
that is suitable for Continual Inference. To do this, we derived
the model updates of the Nystrom-based SDA in a continual ¨
manner, and we proposed two ways tailored to processing
continual stream data for determining the landmarks needed
in the SDA approximation. The resulting model has a linear
computational and memory cost with respect to the number
of input tokens, and achieves faster inference time compared
to competing models, while requiring comparable or lower
memory. Experiments on Audio Classification and Online
Action Detection show that the proposed model leads to a
reduction of up to two orders of magnitude in the number of
operations while retaining similar performance and memory
use compared to competing models.

--

本論文では、継続的な推論に適したNystrom近似に基づくスケールドドットプロダクトアテンションの新しい定式化を導入しました。これを行うために、継続的な方法でNystromベースのSDAのモデル更新を導出し、SDA近似に必要なランドマークを決定するための継続的なストリームデータを処理するために調整された2つの方法を提案しました。結果として得られたモデルは、入力トークンの数に関して線形の計算コストとメモリコストを持ち、競合するモデルと比較してより高速な推論時間を実現し、同等または低いメモリを必要とします。オーディオ分類とオンラインアクション検出に関する実験では、提案されたモデルが競合するモデルと比較して、同様のパフォーマンスとメモリ使用量を維持しながら、最大2桁の操作数の削減につながることが示されています。
-

# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。

We also propose a process for determining appropriate landmarks during the training phase, which can then be used for
processing any received input, avoiding the need to perform
landmark updates during Continual Inference. This approach is
motivated by similar ideas used in approximate kernel-based
learning [36], [37], where landmarks in Nystrom-based ap-
proximation of the kernel matrix are determined by clustering
the training data. However, this approach cannot be directly
applied in our case, as the data transformations performed by
all layers before each of the SDA blocks change at every
training update, leading to different feature spaces in which
the matrices XQ and XK are defined.
---
# translate 703-713
このプロセスは、トレーニングフェーズ中に適切なランドマークを決定するためのプロセスを提案します。これにより、受信した入力を処理する際にランドマークの更新を行う必要がなくなります。このアプローチは、Nystromベースのカーネル行列の近似において、トレーニングデータをクラスタリングすることによってランドマークが決定されるという、近似カーネルベースの学習[36][37]で使用される類似のアイデアに基づいています。ただし、このアプローチは、各SDAブロックの前にすべての層で行われるデータ変換がトレーニングの更新ごとに変化し、行列XQとXKが定義される異なる特徴空間につながるため、私たちのケースには直接適用できません。
---

To address this issue, the training process is divided into two
phases. In the first phase, the model is trained in an end-to-end
manner using continually updated landmarks as described in
Section IV-A. The second phase is divided into two processing
steps. In the first step, the training data is introduced to the
model and the matrices Q and K are calculated for each input
data sample. The Q-tokens corresponding to all training data
are combined to create a dataset which is clustered into m
clusters by applying the m-Means method. The cluster centers
are then used to form the matrix Q˜. The same process is
applied to the K-tokens to form the matrix K˜ . If multiple
SDA heads are used, we compute the landmarks of each head
independently. In the second step, the model is fine-tuned in
an end-to-end manner using the now fixed, landmarks (i.e.,
the matrices Q˜ and K˜ are not updated). When the model is
formed by multiple SDA blocks, the two steps of phase two are
applied sequentially starting from the first block, and keeping
all landmarks of previous SDA blocks fixed in the fine-tuning
step. This leads to gradually determining all landmarks of the
model.
---
この問題に対処するために、トレーニングプロセスは2つのフェーズに分割されます。最初のフェーズでは、セクションIV-Aで説明したように、継続的に更新されたランドマークを使用して、モデルがエンドツーエンドの方法でトレーニングされます。2番目のフェーズは、2つの処理ステップに分かれています。最初のステップでは、トレーニングデータがモデルに導入され、各入力データサンプルについて行列QとKが計算されます。すべてのトレーニングデータに対応するQトークンを組み合わせて、m-Meansメソッドを適用してmクラスタにクラスタリングされたデータセットを作成します。次に、クラスタセンターを使用して行列Q˜を形成します。同じプロセスがKトークンにも適用され、行列K˜が形成されます。複数のSDAヘッドが使用される場合は、それぞれのヘッドのランドマークを独立して計算します。2番目のステップでは、今や固定されたランドマーク(つまり、行列Q˜とK˜は更新されません)を使用して、モデルがエンドツーエンドの方法で微調整されます。モデルが複数のSDAブロックで構成されている場合は、最初のブロックから始めて2つのステップを順次適用し、微調整ステップで前のSDAブロックのすべてのランドマークを固定します。これにより、モデルのすべてのランドマークが徐々に決定されます。
---

 For training, we use a modified version of the noncontinual model with the circular positional encoding
described above and the corresponding landmark selection scheme as described in Sections IV-A and IV-B,
depending on whether continual or fixed landmarks are
used, respectively. We follow this approach as the noncontinual training processes are faster when the entire
sequence is available from the beginning, and both continual and non-continual SDA variants produce identical
results.
---
トレーニングには、上記で説明した円形の位置エンコーディングと、セクションIV-AおよびIV-Bで説明されている対応するランドマーク選択スキームを使用して、非継続的なモデルの修正バージョンを使用します。これは、継続的または固定されたランドマークがそれぞれ使用されるかどうかに応じて行われます。このアプローチに従うのは、非継続的なトレーニングプロセスが最初からシーケンス全体が利用可能な場合により高速であり、継続的および非継続的なSDAバリアントの両方が同一の結果を生成するためです。
---

 The first (n mod m) landmarks
are calculated by using a segment of the token sequence
that has an extra token. The position of these landmarks
is tracked as newer landmarks are included and older
landmarks are discarded, so every new landmark will be
calculated using a segment of the token sequence of the
same size as the landmark it is replacing
---
最初の(n mod m)ランドマークは、追加のトークンを持つトークンシーケンスのセグメントを使用して計算されます。これらのランドマークの位置は、新しいランドマークが含まれ、古いランドマークが破棄されるにつれて追跡されるため、すべての新しいランドマークは、それが置き換えるランドマークと同じサイズのトークンシーケンスのセグメントを使用して計算されます。
---