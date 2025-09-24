# translate English sentence to Japanese
# only translate the sentence to Japanese after the "--" line
# and add a new line after the Japanese sentence

Reservoir computing is a bio-inspired approach for processing time dependent information1–5. A reservoir computer can be decomposed into three parts, see Fig. 1. The “input layer” couples the input signal into a non-linear
dynamical system that constitutes the “reservoir layer”. The internal variables of the dynamical system, also
called “reservoir states”, provide a nonlinear mapping of the input into a high dimensional space. Finally the
time-dependent output of the reservoir is computed in the “output layer” as a linear combination of the internal
variables. The readout weights used to compute this linear combination are optimized so as to minimize the mean
square error between the target and the output signal, leading to a simple and easy training process. On the other
hand, the values of the internal coupling weights within the input layer and within the reservoir layer are not
critical, and can be chosen at random up to some global parameters that are tuned to get the best performance

--

リザーバコンピューティングは、時間依存情報を処理するための生物に触発されたアプローチです1–5。リザーバコンピュータは、図1に示すように、3つの部分に分解できます。「入力層」は、入力信号を「リザーバ層」を構成する非線形動的システムに結合します。動的システムの内部変数は、「リザーバ状態」とも呼ばれ、入力を高次元空間に非線形マッピングします。最後に、リザーバの時間依存出力は、「出力層」で内部変数の線形結合として計算されます。この線形結合を計算するために使用される読み出し重みは、ターゲットと出力信号との平均二乗誤差を最小化するように最適化され、単純で簡単なトレーニングプロセスにつながります。一方、入力層内およびリザーバ層内の内部結合重みの値は重要ではなく、最高のパフォーマンスを得るために調整されるいくつかのグローバルパラメータまでランダムに選択できます。

-

One of the key advantages of reservoir computers is that, because only the output layer is trained, training
algorithms are efficient and rapidly converge to the global optimum. This simplicity enables reservoir computers
to solve a large range of complex tasks on time dependent signals, such as speech recognition6
, nonlinear channel
equalization3,7,8, detection of epileptic seizures9
, robot control10, time series prediction1,3, financial forecasting,
handwriting recognition, etc…, often with state of the art performance. We refer to11–13 for recent reviews.

--
リザーバコンピュータの主な利点の1つは、出力層のみがトレーニングされるため、トレーニングアルゴリズムが効率的であり、グローバル最適解に迅速に収束することです。この単純さにより、リザーバコンピュータは、音声認識6、非線形チャネル等化3,7,8、てんかん発作の検出9、ロボット制御10、時系列予測1,3、金融予測、手書き認識などの時間依存信号に関する幅広い複雑なタスクを解決できます。 、しばしば最先端のパフォーマンスで。最近のレビューについては11–13を参照してください。

-

This section presents the main contribution of the article –
an architecture for integer Echo State Network. The architecture is illustrated in Fig. 2. The proposed intESN is structurally
identical to the the conventional ESN (see Fig. 1) with three
layers of neurons: input (u(n), K neurons), output (y(n), L
neurons), and reservoir (x(n), N neurons). It is important to
note from the beginning that training the readout matrix Wout
for intESN is the same as for the conventional ESN (Section
II-A1).
--
このセクションでは、この記事の主な貢献である整数エコー状態ネットワークのアーキテクチャを紹介します。提案されたintESNは、図2に示すように、従来のESNと構造的に同一です（図1を参照）。3層のニューロンがあります：入力（u（n）、Kニューロン）、出力（y（n）、Lニューロン）、およびリザーバ（x（n）、Nニューロン）。最初から重要な点は、intESNの読み出し行列Woutのトレーニングは、従来のESNと同じであることです（セクションII-A1）。

-
However, other components of intESN differs from the
conventional ESN. First, activations of input and output layers
are projected into the reservoir in the form of bipolar HD
vectors [47] of size N (denoted as u
HD(n) and y
HD(n)).
For problems where input and output data are described by
finite alphabets and each symbol can be treated independently,
the mapping to N-dimensional space is achieved by simply
assigning a random bipolar HD vector to each symbol in the
alphabet and storing them in the item memory [11], [53].
In the case with continuous data (e.g., real numbers), we
quantized the continuous values into a finite alphabet. The
quantization scheme (denoted as Q) and the granularity of
the quantization are problem dependent. Additionally, when
there is a need to preserve similarity between quantization
levels, distance preserving mapping schemes are applied (see,
e.g., [54], [55]), which can preserve, for example, linear or
nonlinear similarity between levels. An example of a discretization and quantization of a continuous signal as well
2
It is not common to do such decoding in RC. Normally, in the scope of
RC a readout matrix is learned. In this article, we follow this standard RC
approach to extracting information back from a reservoir.
as its HD vectors in the item memory is illustrated in Fig. 2.
Continuous values can be also represented in HD vectors by
varying their density. For a recent overview of several mapping
approaches readers are referred to [56]. Also, an example of
applying such mapping is presented in Section IV-A2. Another
feature of intESN is the way the recurrence in the reservoir
is implemented. Rather than a matrix multiply, recurrence
is implemented via the permutation of the reservoir vector.
Note that permutation of a vector can be described in matrix
form, which can play the role of W in intESN. Note that
the spectral radius of this matrix equals one. However, an
efficient implementation of permutation can be achieved for
a special case – cyclic shift (denoted as Sh()). It is important
to note that we have shown in [10] that the recurrent weight
matrix W creates key-value pairs of the input data. Note that
W is chosen randomly and kept fixed, and this always leads
to the same properties. Moreover, there is no advantage of
the fully connected random recurrent weight matrix over the
simple cyclic shift operation for storing the input history. Thus,
the use of the cyclic shift in place of a random recurrent
weight matrix does not limit intESN’s ability to produce
linearly separable representations. Fig. 2 shows the recurrent
connections of neurons in a reservoir with recurrence by cyclic
shift of one position. In this case, vector-matrix multiplication
Wx(n) is equivalent to Sh(x(n), 1).

--
しかし、intESNの他のコンポーネントは、従来のESNとは異なります。まず、入力層と出力層の活性化は、バイポーラHDベクトル[47]の形でリザーバに投影されます（サイズN、u
HD（n）およびy
HD（n））。
入力データと出力データが有限のアルファベットで記述され、各シンボルを独立して扱うことができる問題では、N次元空間へのマッピングは、アルファベット内の各シンボルにランダムなバイポーラHDベクトルを割り当て、それらをアイテムメモリに保存することで達成されます[11]、[53]。連続データ（例：実数）の場合、連続値を有限のアルファベットに量子化しました。量子化スキーム（Qと表記）と量子化の粒度は問題依存です。さらに、量子化レベル間の類似性を保持する必要がある場合、距離保存マッピングスキームが適用されます（例：[54]、[55]を参照）。これは、レベル間の線形または非線形の類似性を保持できます。連続信号の離散化と量子化の例、およびアイテムメモリ内のHDベクトルは図2に示されています。連続値は、その密度を変えることでHDベクトルでも表現できます。いくつかのマッピングアプローチの最近の概要については、[56]を参照してください。また、このようなマッピングを適用する例はセクションIV-A2で示されています。intESNのもう1つの特徴は、リザーバ内の再帰が実装される方法です。行列乗算ではなく、リザーバベクトルの順列によって再帰が実装されます。ベクトルの順列は行列形式で記述できることに注意してください。これはintESNでWの役割を果たすことができます。この行列のスペクトル半径は1に等しいことに注意してください。ただし、順列の効率的な実装は特別なケース – 循環シフト（Sh()と表記）で達成できます。[10]で示したように、再帰重み行列Wは入力データのキーと値のペアを作成します。Wはランダムに選択され、固定されているため、常に同じ特性につながります。さらに、入力履歴を保存するための完全に接続されたランダム再帰重み行列の利点は、単純な循環シフト操作にはありません。したがって、ランダムな再帰重み行列の代わりに循環シフトを使用しても、intESNが線形分離可能な表現を生成する能力は制限されません。図2は、1つの位置の循環シフトによるリザーバ内のニューロンの再帰接続を示しています。この場合、ベクトル-行列乗算Wx(n)はSh(x(n), 1)と同等です。

-

. First, activations of input and output layers
are projected into the reservoir in the form of bipolar HD
vectors [47] of size N
--
最初に、入力層と出力層の活性化は、バイポーラHDベクトル[47]の形でリザーバに投影されます（サイズN）。

-

Reservoir computing (RC), a particular form of recurrent neural network, is under explosive development
due to its exceptional efficacy and high performance in reconstruction or/and prediction of complex physical
systems. However, the mechanism triggering such effective applications of RC is still unclear, awaiting deep
and systematic exploration. Here, combining the delayed embedding theory with the generalized embedding
theory, we rigorously prove that RC is essentially a high dimensional embedding of the original input nonlinear
dynamical system. Thus, using this embedding property, we unify into a universal framework the standard
RC and the time-delayed RC where we novelly introduce time delays only into the network’s output layer,
and we further find a trade-off relation between the time delays and the number of neurons in RC. Based on
these findings, we significantly reduce the RC’s network size and promote its memory capacity in completing
systems reconstruction and prediction. More surprisingly, only using a single-neuron reservoir with time delays
is sometimes sufficient for achieving reconstruction and prediction tasks, while the standard RC of any large
size but without time delay cannot complete them yet.

--
リザーバコンピューティング（RC）は、特定の形式の再帰ニューラルネットワークであり、複雑な物理システムの再構築や予測においてその卓越した効率と高いパフォーマンスにより、爆発的な発展を遂げています。しかし、RCのこのような効果的な応用を引き起こすメカニズムはまだ明らかではなく、深く体系的な探求を待っています。ここでは、遅延埋め込み理論と一般化埋め込み理論を組み合わせて、RCが本質的に元の入力非線形動的システムの高次元埋め込みであることを厳密に証明します。したがって、この埋め込み特性を使用して、標準RCと時間遅延RCを統一し、ネットワークの出力層にのみ新たに時間遅延を導入し、RC内の時間遅延とニューロン数との間のトレードオフ関係をさらに見出します。これらの発見に基づいて、RCのネットワークサイズを大幅に削減し、システムの再構築と予測を完了する際のメモリ容量を向上させます。さらに驚くべきことに、時間遅延付きの単一ニューロンリザーバーのみを使用することで、再構築と予測タスクを達成することができる場合がありますが、時間遅延なしの任意の大きさの標準RCではそれらを完了できません。

-

RCs as different nonlinear dynamical systems and embeddings. (a) A standard RC without time-delay. (b) The generalized
embedding Ψ from the input dynamics to the standard non-delayed
reservoir network and the delayed embedding F from the input dynamics to the delayed reservoir network, which constitute a topological conjugation between the dynamics of the non-delayed reservoir
network and the delayed reservoir network. (c) A time-delayed RC
with a smaller network size in the reservoir layer
--
RCは、異なる非線形動的システムと埋め込みとして機能します。(a) 時間遅延のない標準RC。(b) 入力動力学から標準の非遅延リザーバネットワークへの一般化埋め込みΨと、入力動力学から遅延リザーバネットワークへの遅延埋め込みF。これは、非遅延リザーバネットワークの動力学と遅延リザーバネットワークの間の位相的共役を構成します。(c) リザーバ層でより小さなネットワークサイズを持つ時間遅延RC。

-

The important contributions of this work can be summarized as follows:
(1) A novel IR called the dot-product-based reservoir representation (DPRR), which is computationally efficient and effective for improving the classification accuracy of multivariate time
series data, is proposed.
(2) A fully digital delayed feedback reservoir (DFR) model with DPRR is proposed. The accuracy
of DFR on multivariate time-series classification tasks is thoroughly evaluated. The accuracy
and hardware cost are compared with those of existing machine learning classifiers.
(3) A construct of input-data masking that supports multivariate time-series inputs and suppresses inference accuracy variation is defined.
--
この研究の重要な貢献は次のように要約できます。
(1) 計算効率が高く、多変量時系列データの分類精度を向上させるために効果的な、ドット積ベースのリザーバ表現（DPRR）と呼ばれる新しい情報理論（IR）が提案されました。
(2) DPRRを備えた完全デジタル遅延フィードバックリザーバ（DFR）モデルが提案されました。多変量時系列分類タスクにおけるDFRの精度が徹底的に評価されました。精度とハードウェアコストは、既存の機械学習分類器と比較されます。
(3) 多変量時系列入力をサポートし、推論精度の変動を抑制する入力データマスキングの構造が定義されました。

-

Generally, computations are conducted with floating point numbers, which are an exponential representation, and can represent
a wide range of numbers. A circuit using floating point numbers
is more complex as it requires many FPGA resources. In contrast, although the fixed-point representation can only represent
a narrow range of numbers, the circuit resources is less complex
compared with that using the floating point.
--
一般的に、計算は浮動小数点数で行われます。これは指数表現であり、幅広い数値を表現できます。浮動小数点数を使用する回路は、FPGAリソースを多く必要とするため、より複雑です。対照的に、固定小数点表現は狭い範囲の数値しか表現できませんが、浮動小数点を使用する回路と比較して、回路リソースはそれほど複雑ではありません。
-

One way to reduce the complexity of a circuit is using quantized
values that are able to simplify the computation while maintaining
its accuracy [5].
Therefore, we calculated the outputs of the reservoir layer
[Equation (1)] using quantized weights. Generally, the weights
of input and reservoir layers are real numbers resulting in
several DSPs to compute real number multiplications. Therefore,
we transformed the real valued weights to ternary values: 0 or ±1.
Furthermore, the accuracy by using this quantization for both
training and prediction mode are maintained.
The circuit of the neuron is shown in Figure 2. Where n is the
number of reservoir’s neurons. un
 and wn
 are inputs and weights of
input and reservoir layers, respectively, and m is the bit width of the
input data. Furthermore, the circuit is able to calculate accumulate
operations using only AND and OR operations.
We have verified the accuracy of a quantized model and the conventional model. The task carried out to evaluate their performance
was NARMA10 [6] with equations as follows:
--
回路の複雑さを減らす方法の1つは、計算を簡素化しながら精度を維持できる量子化された値を使用することです[5]。
したがって、リザーバ層の出力[式（1）]を量子化された重みを使用して計算しました。一般に、入力層とリザーバ層の重みは実数であり、実数乗算を計算するためにいくつかのDSPが必要です。したがって、実数値の重みを三元値：0または±1に変換しました。さらに、この量子化をトレーニングモードと予測モードの両方で使用することによる精度は維持されます。
ニューロンの回路は図2に示されています。ここで、nはリザーバのニューロンの数です。un
とwn
は、入力層とリザーバ層の入力と重みであり、mは入力データのビット幅です。さらに、この回路はANDおよびOR操作のみを使用して累積操作を計算できます。
NARMA10のパフォーマンスを評価するために実行されたタスクは、次の式を持つNARMA10 [6]でした：

-
where uk
 and yk
 are input and output at time k. a, b, g, d, m, s are
hyper parameters which we set to value (0.3, 0.05, 1.5, 0.1, 1, 0.5).
vk
 is the random number from 0 to 1. Training data contains 4000
time steps and test data contains 300 time steps, but we used only
the last 200 time steps data.
Figure 3 shows the prediction of 100–200 time steps of the quantized model where the number of reservoir neurons was 1000.
The black line represents the supervised signal and the blue line
represents output of the quantized model. The quantized model
was able to reproduce NARMA10. Figure 4 shows the MSE of the
supervised signal and outputs of each model, with varying in the
number of neurons in the reservoir. The accuracy of the quantized
model was similar to conventional model.
--
uk
とyk
は、時間kにおける入力と出力です。a、b、g、d、m、sはハイパーパラメータであり、値（0.3、0.05、1.5、0.1、1、0.5）に設定しました。
vk
は0から1のランダムな数です。トレーニングデータには4000タイムステップが含まれ、テストデータには300タイムステップが含まれていますが、最後の200タイムステップのデータのみを使用しました。
図3は、リザーバニューロンの数が1000の量子化モデルの100〜200タイムステップの予測を示しています。黒い線は監視信号を表し、青い線は量子化モデルの出力を表します。量子化モデルはNARMA10を再現できました。図4は、リザーバ内のニューロン数を変化させた各モデルの監視信号と出力のMSEを示しています。量子化モデルの精度は従来のモデルと同様でした。
-

As shown in Figure 5, general product–sum operations can be represented by a tree structure. Using this representation, the number
of adders and multipliers increases with the number of neurons. 
--
図5に示すように、一般的な積和演算はツリー構造で表現できます。この表現を使用すると、ニューロンの数が増えるにつれて加算器と乗算器の数が増加します。

-

Therefore, in this research, we sequentially calculated the product–
sum of the output layer. Figure 6 illustrates the product–sum operation by the proposed method, where Ai
 is an intermediate variable
that temporarily stores the accumulate value. As this method consists of only a single adder, multiplier, and register per neuron in
the output layer, the complexity of the circuit is reduced.

--

したがって、この研究では、出力層の積和を順次計算しました。図6は、提案された方法による積和演算を示しています。ここで、Ai
は、累積値を一時的に保存する中間変数です。この方法は、出力層のニューロンごとに単一の加算器、乗算器、およびレジスタのみで構成されているため、回路の複雑さが減少します。

-

As shown in Figure 7, the conventional model used two pipeline. In
process 1 and 2, the reservoir module calculates the state of a single
neuron in reservoir layer and stores it in memory. In process 3, the
reservoir module repeats the process 1 and 2 for the rest of reservoir
neurons. In process 4, an output module calculates the output of a
single neuron in output layer by using tree structure. In process 5,
the output module repeats process 4 for the rest of output neurons.
Figure 8 illustrates the circuits architecture of the proposed model.
We implement the sequential structure of the product–sum circuit (as in Figure 6) in parallel for the output layer. Therefore, the
proposed circuit is able to calculate a single neuron of reservoir
layer and output layer simultaneously in process 4. As a result, the
proposed model processes more efficiency than the conventional
model. Table 1 shows the comparison between the conventional
model and the proposed model.
--
図7に示すように、従来のモデルは2つのパイプラインを使用しました。プロセス1と2では、リザーバモジュールがリザーバ層の単一ニューロンの状態を計算し、それをメモリに保存します。プロセス3では、リザーバモジュールが残りのリザーバニューロンに対してプロセス1と2を繰り返します。プロセス4では、出力モジュールがツリー構造を使用して出力層の単一ニューロンの出力を計算します。プロセス5では、出力モジュールが残りの出力ニューロンに対してプロセス4を繰り返します。図8は、提案されたモデルの回路アーキテクチャを示しています。出力層の積和回路（図6のような）の順次構造を並列に実装します。したがって、提案された回路は、プロセス4でリザーバ層と出力層の単一ニューロンを同時に計算できます。その結果、提案されたモデルは従来のモデルよりも効率的に処理します。表1は、従来のモデルと提案されたモデルの比較を示しています。
-

In the experiment, we have created two types of circuits in order
to verify the effectiveness of the proposed circuit and compared
its calculation speed with those of the other devices. The task to
evaluate their performance is the prediction problem of sine and
cosine waves. The number of neurons in the input, reservoir and
the output layers were 2, 100, and 2, respectively, and the prediction
was computed in an FPGA. The target device is a Zynq UltraScale+
MPSoC ZCU102. Furthermore, the experiment was conducted
with an operating frequency of 200 MHz and a data width of 32-bits
operations [7]. Table 2 shows experimental conditions.
--
実験では、提案された回路の有効性を検証するために2種類の回路を作成し、その計算速度を他のデバイスと比較しました。パフォーマンスを評価するタスクは、正弦波と余弦波の予測問題です。入力層、リザーバ層、および出力層のニューロン数はそれぞれ2、100、および2であり、予測はFPGAで計算されました。ターゲットデバイスはZynq UltraScale+ MPSoC ZCU102です。さらに、実験は200 MHzの動作周波数と32ビットのデータ幅で実施されました[7]。表2は実験条件を示しています。
-

We were able to successfully adapt the circuit to enhance ESN computation in the FPGA. As a result, high-speed computation was possible while the circuit resources were reduced. To achieve this,
fixed-point computation, quantification of weights, and sequential
product–sum computation techniques were used.
In the future, it is expected to apply the proposed circuit and methods to embedded systems such as automobiles and robots.
--
回路をFPGAでのESN計算の強化に適応させることに成功しました。その結果、回路リソースが削減されながら、高速計算が可能になりました。これを達成するために、固定小数点計算、重みの量子化、および順次積和計算技術が使用されました。
将来的には、提案された回路と方法を自動車やロボットなどの組み込みシステムに適用することが期待されています。
-

Figure 9 shows the prediction of the conventional and proposed
circuits. The black, blue, and red lines represent the supervised
signal, prediction of the conventional circuit and prediction of the
proposed circuit, respectively. Both circuits were able to reproduce sine and cosine waves. Tables 3 and 4 shows the utilization
of resources for the conventional and proposed circuit, respectively. The proposed method was able to reduce the overall use
of resources approximately 50%. Table 5 shows the comparison
between electric energy of conventional circuit and proposed circuit. The proposed method reduced the electric energy consumption by approximately 80% compared with the conventional one.
Table 6 shows a comparison between the computation speed of the
FPGA and other devices. The proposed circuit was approximately
25 and 340 times faster than a desktop CPU and embedded CPU,
respectively. 

--

図9は、従来の回路と提案された回路の予測を示しています。黒、青、赤の線は、それぞれ監視信号、従来の回路の予測、および提案された回路の予測を表しています。両方の回路は正弦波と余弦波を再現できました。表3と4は、それぞれ従来の回路と提案された回路のリソース使用率を示しています。提案された方法は、リソースの全体的な使用量を約50％削減できました。表5は、従来の回路と提案された回路の電気エネルギーの比較を示しています。提案された方法は、従来のものと比較して電気エネルギー消費を約80％削減しました。表6は、FPGAと他のデバイスの計算速度の比較を示しています。提案された回路は、デスクトップCPUと組み込みCPUよりもそれぞれ約25倍および340倍速くなりました。

-

Neural networks are highly expected to be applied into embedded
systems such as robots and automobiles. However, Deep Neural
Networks (DNNs) [1] require high computational power because
a lot of accumulate operations are being processed using them.
Generally, graphics processing units are used to accelerate these
computations; however, as their power consumption is high, implementing embedded systems using them is difficult due to a power
limit. To mitigate this problem, we have implemented DNNs into
hardware such as Field Programmable Gate Arrays (FPGAs), realizing high-speed calculation with low power consumption.
In this paper, we have implemented an Echo State Network (ESN)
[2], a kind of Reservoir Computing (RC) into an FPGA. An RC is a
Recurrent Neural Network (RNN) model in which only the weights
of an output layer are defined in the training step. ESNs are able to
learn time-series data faster than general RNNs such as Long Shortterm Memory (LSTM). In ESNs, a lot of accumulate operations of
input data and weights are executed, however, there are limitations
of FPGA resources such as Loot Up Table (LUT), Flip Flop (FF) and
Digital Signal Processor (DSP). As a result, we have modified the
algorithms and architectures of ESNs. Furthermore, we implement
the proposed hardware-oriented algorithms into the FPGA and
show the effectiveness of the proposed methods by comparing the
proposed circuit with other.

--

ニューラルネットワークは、ロボットや自動車などの組み込みシステムへの適用が期待されています。しかし、深層ニューラルネットワーク（DNN）[1]は、多くの累積操作が処理されるため、高い計算能力を必要とします。一般的に、これらの計算を加速するためにグラフィックス処理ユニットが使用されますが、その消費電力が高いため、組み込みシステムでの実装は電力制限のために困難です。この問題を軽減するために、フィールドプログラマブルゲートアレイ（FPGA）などのハードウェアにDNNを実装し、低消費電力で高速計算を実現しました。
この論文では、エコー状態ネットワーク（ESN）[2]、リザーバコンピューティング（RC）の一種をFPGAに実装しました。RCは、出力層の重みのみがトレーニングステップで定義される再帰ニューラルネットワーク（RNN）モデルです。ESNは、長短期記憶（LSTM）などの一般的なRNNよりも時系列データを高速に学習できます。ESNでは、入力データと重みの多くの累積操作が実行されますが、ルックアップテーブル（LUT）、フリップフロップ（FF）、デジタル信号プロセッサ（DSP）などのFPGAリソースには制限があります。その結果、ESNのアルゴリズムとアーキテクチャを変更しました。さらに、提案されたハードウェア指向のアルゴリズムをFPGAに実装し、提案された回路と他の回路を比較することで提案された方法の有効性を示します。

-

Therefore, we calculated the outputs of the reservoir layer
[Equation (1)] using quantized weights. Generally, the weights
of input and reservoir layers are real numbers resulting in
several DSPs to compute real number multiplications. Therefore,
we transformed the real valued weights to ternary values: 0 or ±1.
Furthermore, the accuracy by using this quantization for both
training and prediction mode are maintained.
The circuit of the neuron is shown in Figure 2. Where n is the
number of reservoir’s neurons. un
 and wn
 are inputs and weights of
input and reservoir layers, respectively, and m is the bit width of the
input data. Furthermore, the circuit is able to calculate accumulate
operations using only AND and OR operations.

--

したがって、リザーバ層の出力[式（1）]を量子化された重みを使用して計算しました。一般に、入力層とリザーバ層の重みは実数であり、実数乗算を計算するためにいくつかのDSPが必要です。したがって、実数値の重みを三元値：0または±1に変換しました。さらに、この量子化をトレーニングモードと予測モードの両方で使用することによる精度は維持されます。
ニューロンの回路は図2に示されています。ここで、nはリザーバのニューロンの数です。un
とwn
は、入力層とリザーバ層の入力と重みであり、mは入力データのビット幅です。さらに、この回路はANDおよびOR操作のみを使用して累積操作を計算できます。
--

Figure 3 shows the prediction of 100–200 time steps of the quantized model where the number of reservoir neurons was 1000.
The black line represents the supervised signal and the blue line
represents output of the quantized model. The quantized model
was able to reproduce NARMA10. Figure 4 shows the MSE of the
supervised signal and outputs of each model, with varying in the
number of neurons in the reservoir. The accuracy of the quantized
model was similar to conventional model.
--
図3は、リザーバニューロンの数が1000の量子化モデルの100〜200タイムステップの予測を示しています。黒い線は監視信号を表し、青い線は量子化モデルの出力を表します。量子化モデルはNARMA10を再現できました。図4は、リザーバ内のニューロン数を変化させた各モデルの監視信号と出力のMSEを示しています。量子化モデルの精度は従来のモデルと同様でした。

-

As shown in Figure 5, general product–sum operations can be represented by a tree structure. Using this representation, the number
of adders and multipliers increases with the number of neurons. 
--
図5に示すように、一般的な積和演算はツリー構造で表現できます。この表現を使用すると、ニューロンの数が増えるにつれて加算器と乗算器の数が増加します。

-

Therefore, in this research, we sequentially calculated the product–
sum of the output layer. Figure 6 illustrates the product–sum operation by the proposed method, where Ai is an intermediate variable
that temporarily stores the accumulate value. As this method consists of only a single adder, multiplier, and register per neuron in
the output layer, the complexity of the circuit is reduced.

--

したがって、この研究では、出力層の積和を順次計算しました。図6は、提案された方法による積和演算を示しています。ここで、Aiは累積値を一時的に保存する中間変数です。この方法は、出力層のニューロンごとに単一の加算器、乗算器、およびレジスタのみで構成されているため、回路の複雑さが減少します。

-
