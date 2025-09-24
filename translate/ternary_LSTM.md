# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。

According to the LSTM formula in the “Background” section, forwarding weights (Wx), recurrent
weights (Wh), and the bias (b) vectors are learned
during training. For example, the forwarding, and
recurrent weight matrices and bias vectors for the
gate i are denoted as Wix , Wih, and bi
, respectively.
Since in a full precision LSTM, all inputs and all mentioned parameters are full precision, a significant
amount of memory and computation is required. In
this work, we have proposed three architectures for
ternary LSTM called FaCT-LSTM1-3 that eliminate
conventional computation-intensive multiplication
operations with negligible accuracy loss.
In the proposed architectures, weights, activations, and the output of the activation functions
are transformed to 2-bit ternary values. To obtain
the efficient coding of the ternary values, we have
performed a comprehensive exploration to find all
possible 2-bit codings for {−1, 0, 1} values. Since two
bits have four representations, one of the {−1, 0, 1}
values has the chance of having two codes. Hence,
we first determine which value of {−1, 0, 1} is a better candidate for having two representations to find
the efficient coding that can minimize logic gates for
multiplication operations. The results of our extensive explorations demonstrated that the optimal
coding is when we have two representations for 0.
Considering two representations for 0, 8 codings
out of 12 total possible codings, require only two
gates for 2-bit multiplication operation. Moreover,
by deploying two representations for 0, 12 outputs in the truth table of the 2-bit multiplication are 0.
Hence, we can also benefit from zero skippings in
this encoding. The proposed encoding for ternary
values is shown in Figure 2b. Furthermore, to reduce
the effect of quantization error, we have considered
a distinguished scaling factor for each weight, bias,
and input. It should be noted that these scaling factors are learned during the training phase.
---
LSTMの式に基づいて、フォワーディング重み（Wx）、再帰重み（Wh）、およびバイアス（b）ベクトルはトレーニング中に学習されます。例えば、ゲートiのフォワーディングおよび再帰重み行列とバイアスベクトルは、それぞれWix、Wih、およびbiと表されます。フルプレシジョンLSTMでは、すべての入力と前述のパラメータがフルプレシジョンであるため、大量のメモリと計算が必要です。本研究では、従来の計算集約型乗算操作をほとんど精度を損なうことなく排除する3つの三値LSTMアーキテクチャ（FaCT-LSTM1-3）を提案しました。
提案されたアーキテクチャでは、重み、活性化、および活性化関数の出力が2ビットの三値に変換されます。三値の効率的なコーディングを得るために、{−1、0、1}の値に対するすべての可能な2ビットコーディングを見つけるための包括的な探索を行いました。2ビットには4つの表現があるため、{−1、0、1}の値のうち1つは2つのコードを持つ可能性があります。したがって、乗算操作の論理ゲートを最小化できる効率的なコーディングを見つけるために、どの{−1、0、1}の値が2つの表現を持つのに適しているかを最初に決定します。広範な探索の結果、最適なコーディングは0に対して2つの表現がある場合であることが示されました。0に対して2つの表現を考慮すると、12個の可能なコーディングのうち8個は2ビット乗算操作に対してわずか2つのゲートしか必要としません。さらに、0に対して2つの表現を使用することで、2ビット乗算の真理値表では12個の出力が0になります。したがって、このコーディングではゼロスキップも利用できます。三値値に対する提案されたコーディングは図2bに示されています。さらに、量子化誤差の影響を減らすために、各重み、バイアス、および入力に対して区別されたスケーリング係数を考慮しました。これらのスケーリング係数はトレーニングフェーズ中に学習されることに注意してください。
-

Functional near-infrared spectroscopy (fNIRS) is a
non-invasive, low-cost method used to study the brain’s blood
flow pattern. Such patterns can enable us to classify performed
by a subject. In recent research, most classification systems use
traditional machine learning algorithms for the classification of
tasks. These methods, which are easier to implement, usually
suffer from low accuracy. Further, a complex pre-processing
phase is required for data preparation before implementing
traditional machine learning methods. The proposed system uses
a Bi-Directional LSTM based deep learning architecture for task
classification, including mental arithmetic, motor imagery, and
idle state using fNIRS data. Further, this system will require less
pre-processing than the traditional approach, saving time and
computational resources while obtaining an accuracy of 81.48%,
which is considerably higher than the accuracy obtained using
conventional machine learning algorithms for the same data set.
---
機能的近赤外分光法（fNIRS）は、脳の血流パターンを研究するために使用される非侵襲的で低コストの方法です。このようなパターンにより、被験者が実行したタスクを分類することが可能になります。最近の研究では、ほとんどの分類システムがタスクの分類に従来の機械学習アルゴリズムを使用しています。これらの方法は実装が容易ですが、通常は精度が低いという欠点があります。さらに、従来の機械学習方法を実装する前に、データ準備のために複雑な前処理フェーズが必要です。提案されたシステムは、fNIRSデータを使用して、メンタルアリスメティック、モーターイメージング、およびアイドル状態を含むタスク分類のために双方向LSTMベースの深層学習アーキテクチャを使用します。さらに、このシステムは従来のアプローチよりも前処理が少なくて済み、時間と計算リソースを節約しながら、同じデータセットで従来の機械学習アルゴリズムを使用した場合よりもかなり高い精度81.48%を達成します。

-
For this paper, a deep learning model using LSTM was
developed for ternary classification. This data was used to
compare models that use different feature sets and even raw data. Deep Learning classifiers are powerful tools that
can handle data with complex nonlinear relationships. The
classification accuracy into varying levels of data preparation
can help assess how much data pre-processing is required for
the deep learning classifiers.
---
この論文では、三値分類のためにLSTMを使用した深層学習モデルが開発されました。このデータは、異なる特徴セットや生データを使用するモデルを比較するために使用されました。深層学習分類器は、複雑な非線形関係を持つデータを処理できる強力なツールです。データ準備のレベルに応じた分類精度は、深層学習分類器に必要なデータ前処理の量を評価するのに役立ちます。
-

Recurrent neural networks (RNNs) have shown excellent performance in processing sequence data. However, they are both complex and memory intensive due
to their recursive nature. These limitations make RNNs difficult to embed on
mobile devices requiring real-time processes with limited hardware resources. To
address the above issues, we introduce a method that can learn binary and ternary
weights during the training phase to facilitate hardware implementations of RNNs.
As a result, using this approach replaces all multiply-accumulate operations by
simple accumulations, bringing significant benefits to custom hardware in terms
of silicon area and power consumption. On the software side, we evaluate the
performance (in terms of accuracy) of our method using long short-term memories
(LSTMs) and gated recurrent units (GRUs) on various sequential models including
sequence classification and language modeling. We demonstrate that our method
achieves competitive results on the aforementioned tasks while using binary/ternary
weights during the runtime. On the hardware side, we present custom hardware
for accelerating the recurrent computations of LSTMs with binary/ternary weights.
Ultimately, we show that LSTMs with binary/ternary weights can achieve up to
12× memory saving and 10× inference speedup compared to the full-precision
hardware implementation design.
---
再帰型ニューラルネットワーク（RNN）は、シーケンスデータの処理において優れた性能を示しています。しかし、再帰的な性質のために複雑でメモリ集約的であり、これらの制限により、RNNはリアルタイムプロセスを必要とするモバイルデバイスに埋め込むことが困難です。これらの問題に対処するために、RNNのハードウェア実装を容易にするために、トレーニングフェーズ中にバイナリおよび三値重みを学習できる方法を導入します。その結果、このアプローチを使用すると、すべての乗算累積操作が単純な累積に置き換えられ、シリコン面積と消費電力の点でカスタムハードウェアに大きな利点をもたらします。ソフトウェア側では、長短期記憶（LSTM）およびゲート付き再帰ユニット（GRU）を使用して、シーケンス分類や言語モデリングなどのさまざまなシーケンシャルモデルでの精度に関して我々の方法の性能を評価します。我々の方法は、ランタイム中にバイナリ/三値重みを使用しながら、前述のタスクで競争力のある結果を達成できることを示します。ハードウェア側では、バイナリ/三値重みを持つLSTMの再帰計算を加速するためのカスタムハードウェアを提示します。最終的に、バイナリ/三値重みを持つLSTMは、フルプレシジョンハードウェア実装設計と比較して最大12倍のメモリ節約と10倍の推論速度向上を達成できることを示します。

-
In the first architecture, we only ternarize all weights and biases of the LSTM, including
forwarding weight (Wx), recurrent weights (Wh), and
bias weights (b) in (1)−(4) to {−1,0,1} values according to our ternarization function in (12). We use the
encoding shown in Figure 2b for ternary weights, and
the inputs are full precision. In this case, the multiplication and accumulation operations are replaced
with simple accumulations and multiplexers. Using
the encoding of Figure 2b for ternary weights, multiplication operation can be implemented with a 4 to
1 multiplexer that is controlled by the 2-bit ternary
weight (Wi) as its select signal and the inputs of the
multiplexer are fed with four full precision values
including Xi,−Xi, and 0. When Wi = 01, the multiplexer
passes Xi and when Wi =11, the multiplexer passes
−Xi, and when Wi = 00 or Wi =10 the multiplexer passes
0 then the output of the multiplexer is accumulated to
its previous outputs for implementing accumulation.
In FaCT-LSTM1, the proposed multiplier for performing dot products between full precision inputs and ternary weights is called MUXSUM (Figure 2a). Although
the computation-intensive multiplications are eliminated in this architecture, the inputs are still full precision, and it still requires full precision accumulation
operations which challenge the resource-limited
health care applications.
---
最初のアーキテクチャでは、LSTMのすべての重みとバイアス（フォワーディング重み（Wx）、再帰重み（Wh）、および式（1）〜（4）のバイアス重み（b））を三値化し、(12)の三値化関数に従って{−1,0,1}の値に変換します。三値重みに対して図2bに示されているエンコーディングを使用し、入力はフルプレシジョンです。この場合、乗算および累積操作は単純な累積とマルチプレクサに置き換えられます。図2bの三値重みのエンコーディングを使用すると、乗算操作は4対1のマルチプレクサで実装でき、これは2ビットの三値重み（Wi）が選択信号として制御され、マルチプレクサの入力はXi、−Xi、および0を含む4つのフルプレシジョン値で供給されます。Wi = 01の場合、マルチプレクサはXiを通過し、Wi = 11の場合は−Xiを通過し、Wi = 00またはWi = 10の場合は0を通過します。その後、マルチプレクサの出力は前の出力に累積されて累積を実装します。FaCT-LSTM1では、フルプレシジョン入力と三値重みとの間でドット積を実行するために提案された乗算器はMUXSUMと呼ばれます（図2a）。計算集約型の乗算がこのアーキテクチャでは排除されていますが、入力は依然としてフルプレシジョンであり、リソース制限されたヘルスケアアプリケーションに挑戦するフルプレシジョン累積操作が必要です。
-

In this section, we review the basic LSTM
approach, earlier presented in [9]. The LSTM architecture is composed of several recurrently connected
“memory cells.” An LSTM cell is shown in Figure 1.
Each cell is composed of three multiplicative gating
connections, namely, input (it), forget (ft), and output (ot) gates and the function of each gate can be
interpreted as write, reset, and read operations, concerning the internal cell state. The gates in a memory
cell facilitate keeping and accessing the internal cell
states over long periods of time. The LSTM output
would depend on all previous inputs. Previous information is neither completely discarded nor completely carried over to the current state. Instead, the
influence of the previous information on the current
state is carefully controlled through the gate signals
---
このセクションでは、基本的なLSTMアプローチについてレビューします。これは以前に[9]で提示されました。LSTMアーキテクチャは、いくつかの再帰的に接続された「メモリセル」で構成されています。LSTMセルは図1に示されています。各セルは、入力（it）、忘却（ft）、および出力（ot）ゲートの3つの乗算ゲート接続で構成されており、各ゲートの機能は内部セル状態に関する書き込み、リセット、および読み取り操作として解釈できます。メモリセル内のゲートは、内部セル状態を長期間保持およびアクセスすることを容易にします。LSTMの出力はすべての前の入力に依存します。前の情報は完全には破棄されず、現在の状態に完全には引き継がれません。代わりに、前の情報が現在の状態に与える影響は、ゲート信号を通じて慎重に制御されます。

--
In this architecture, both of the
weights and activations in (1)−(4) and the output of
the activation functions in (5)−(8) are transformed
to 2-bit ternary values. Based on the optimized coding found in our exploration (Figure 2b), we propose
a simple architecture instead of the MAC operation which is called LogicPopcount (Figure 2e). In
other words, we deploy two gates (xnor and and as
shown in Figure 2c) for multiplication and also we
propose popcount unit for accumulation of multiplication results in convolution operations (Figure 2e).
Figure 2e shows the popcount unit in detail, which
consists of two bitcount and one adder modules.
Considering the proposed coding, bitcount1 counts
the number of one values in the lower bit position
of the 2-bit multiplication output, which denotes the
total number of +1 and −1 values (N). The bitcount2
counts the number of −1 values (n). For this purpose,
the results of ANDing MSB and LSB bits are fed to
bitcount2 (for encoding −1). Finally, the outputs
of both bitcounts are fed to the adder to compute
N−2n as the final result of popcount which represents
the sum of all 2-bit inputs. It should be mentioned
that the logic gates would be different based on the
encoding of {−1,0,1}. 
---
このアーキテクチャでは、(1)−(4)の重みと活性化、および(5)−(8)の活性化関数の出力が2ビットの三値に変換されます。探索で見つけた最適化されたコーディング（図2b）に基づいて、MAC操作の代わりにLogicPopcountと呼ばれるシンプルなアーキテクチャを提案します。つまり、乗算には2つのゲート（図2cに示すxnorとand）を使用し、畳み込み操作の乗算結果を累積するためにpopcountユニットを提案します（図2e）。図2eはpopcountユニットの詳細を示しており、2つのbitcountモジュールと1つの加算器モジュールで構成されています。提案されたコーディングを考慮すると、bitcount1は2ビット乗算出力の下位ビット位置にある1の値の数をカウントし、+1と−1の値の合計数（N）を示します。bitcount2は−1の値の数（n）をカウントします。この目的のために、MSBとLSBビットをANDingした結果がbitcount2に供給されます（−1のエンコード用）。最後に、両方のbitcountの出力が加算器に供給され、popcountの最終結果としてN−2nが計算されます。これはすべての2ビット入力の合計を表します。{−1,0,1}のエンコードによって論理ゲートは異なることに注意してください。
--

Quantization is a promising technique to reduce the model size, memory footprint,
and computational cost of neural networks for the employment on embedded
devices with limited resources. Although quantization has achieved impressive
success in convolutional neural networks (CNNs), it still suffers from large accuracy
degradation on recurrent neural networks (RNNs), especially in the extremely lowbit cases. In this paper, we first investigate the accuracy degradation of RNNs
under different quantization schemes and visualize the distribution of tensor values
in the full precision models. Our observation reveals that due to the different
distributions of weights and activations, different quantization methods should be
used for each part. Accordingly, we propose HitNet, a hybrid ternary RNN, which
bridges the accuracy gap between the full precision model and the quantized model
with ternary weights and activations. In HitNet, we develop a hybrid quantization
method to quantize weights and activations. Moreover, we introduce a sloping
factor into the activation functions to address the error-sensitive problem, further
closing the mentioned accuracy gap. We test our method on typical RNN models,
such as Long-Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU).
Overall, HitNet can quantize RNN models into ternary values of {-1, 0, 1} and
significantly outperform the state-of-the-art methods towards extremely quantized
RNNs. Specifically, we improve the perplexity per word (PPW) of a ternary LSTM
on Penn Tree Bank (PTB) corpus from 126 to 110.3 and a ternary GRU from 142
to 113.5.
---
量子化は、リソースが限られた組み込みデバイスでの使用のために、ニューラルネットワークのモデルサイズ、メモリフットプリント、および計算コストを削減する有望な手法です。量子化は畳み込みニューラルネットワーク（CNN）で印象的な成功を収めていますが、再帰型ニューラルネットワーク（RNN）では特に極端な低ビットケースで大きな精度低下に悩まされています。本論文では、まず異なる量子化スキーム下でのRNNの精度低下を調査し、フルプレシジョンモデルにおけるテンソル値の分布を可視化します。我々の観察により、重みと活性化の分布が異なるため、各部分に対して異なる量子化方法を使用する必要があることが明らかになりました。それに応じて、HitNetというハイブリッド三値RNNを提案します。これは、フルプレシジョンモデルと三値重みおよび活性化を持つ量子化モデルとの間の精度ギャップを埋めるものです。HitNetでは、重みと活性化を量子化するためのハイブリッド量子化方法を開発します。さらに、エラー感受性問題に対処するために活性化関数に傾斜係数を導入し、前述の精度ギャップをさらに縮小します。Long-Short-Term Memory（LSTM）やGated Recurrent Unit（GRU）などの典型的なRNNモデルで我々の方法をテストします。全体として、HitNetはRNNモデルを{-1, 0, 1}の三値に量子化でき、極端に量子化されたRNNに対して最先端の方法よりも大幅に優れた性能を発揮します。具体的には、Penn Tree Bank（PTB）コーパス上で三値LSTMの単語あたりの困惑度（PPW）を126から110
.3に、三値GRUを142から113.5に改善します。
--

memory (LSTM), as the
most popular and effective recurrent neural network (RNN) model, is proved to be successful for
many domains including health care applications,
speech recognition, and language modeling. To further improve the LSTM model accuracy, researchers
usually introduce large models, resulting in practical
hardness, especially on embedded and wearable
devices due to inferior resources for computation and
memory. Model compression is an effective method to alleviate the aforementioned problems. Quantization is a popular compression technique that is
quite promising to reduce
the complexity of deep
neural networks (DNNs)
[1]. Ternarization, as a
particular case of quantization, efficiently compresses
DNN models and mainly provides three benefits: 
---
このセクションでは、RNNの中で最も人気があり効果的なモデルである長短期記憶（LSTM）について説明します。LSTMは、ヘルスケアアプリケーション、音声認識、言語モデリングなど、多くのドメインで成功を収めています。LSTMモデルの精度をさらに向上させるために、研究者は通常、大規模なモデルを導入しますが、これは特に計算とメモリのリソースが劣る組み込みおよびウェアラブルデバイスでは実用的な困難を引き起こします。モデル圧縮は、前述の問題を緩和するための効果的な方法です。量子化は、深層ニューラルネットワーク（DNN）の複雑さを削減するために非常に有望な人気のある圧縮技術です[1]。三値化は量子化の特別なケースとして、DNNモデルを効率的に圧縮し、主に次の3つの利点を提供します：

-

• It converts the 32-bit floating-point parameters
into 2-bit ternary format (i.e., −1, 0, +1), which
can significantly reduce the model size.
• Ternary weights substitute compute-intensive
multiply–accumulate (MAC) operations with
hardware-friendly multiplexer and addition/
subtraction operations and hence, significantly
reduce the inference latency.
• Ternary parameters with zero values intrinsically
prune network connections; thus, the computations related to zero weights can be simply skipped.
---
• 32ビットの浮動小数点パラメータを2ビットの三値形式（つまり、−1、0、+1）に変換し、モデルサイズを大幅に削減できます。
• 三値重みは、計算集約型の乗算-累積（MAC）操作をハードウェアに優しいマルチプレクサと加算/減算操作に置き換え、推論レイテンシを大幅に削減します。
• ゼロ値の三値パラメータは、ネットワーク接続を本質的にプルーニングします。したがって、ゼロ重みに関連する計算は単純にスキップできます。
-

Quantization achieved outstanding success and
satisfied the accuracy requirements on convolutional
neural networks (CNNs) even with only 2-bit ternary
weights and activations [2]. However, accuracy results
on quantized RNNs are still not satisfactory [3]–[5].
Using extreme low-bit precisions in the quantized
RNN models may lead to large accuracy loss that can
hinder the use of RNNs (especially LSTM networks) in
some applications such as health care signal classification. This motivated us to propose a technique to
bridge the huge accuracy gap in aggressive quantization on LSTMs. To the best of our knowledge, it is the
first work on ternarizing the LSTM network in health
care applications. Our proposed method employs
RNNs because the waveforms are naturally fit to be
processed by this type of neural network. We have
previously attempted to cope with the computational
and resource limitations by using 5-bit values for both
weights and inputs and also the power of two scaling factors for electroencephalography (EEG) signal
classification which have achieved accuracies close
to the full precision [5]. Then in [6], we have further
reduced the bits to 3 bits for both inputs and weights.
However, to ensure the accuracy close to the full
precision, we have deployed different full precision
scaling factors for each level of binarization, which
require compute-intensive MAC operations [6]. 
---
量子化は、特に2ビットの三値重みおよび活性化のみで、畳み込みニューラルネットワーク（CNN）で優れた成功を収め、精度要件を満たしました[2]。しかし、量子化されたRNNの精度結果は依然として満足のいくものではありません[3]–[5]。量子化されたRNNモデルで極端な低ビット精度を使用すると、大きな精度損失が生じ、RNN（特にLSTMネットワーク）の一部のアプリケーション、例えばヘルスケア信号分類での使用が妨げられる可能性があります。これにより、LSTMの積極的な量子化における大きな精度ギャップを埋める技術を提案する動機が生まれました。私たちの知る限り、これはヘルスケアアプリケーションにおけるLSTMネットワークの三値化に関する最初の研究です。提案された方法では、波形がこのタイプのニューラルネットワークによって処理されるのに自然に適しているため、RNNを使用しています。以前は、重みと入力の両方に5ビット値を使用し、脳波（EEG）信号分類のために2のべき乗スケーリング係数を使用して計算とリソースの制限に対処し、フルプレシジョンに近い精度を達成しました[5]。次に[6]では、入力と重みの両方でビット数を3ビットにさらに削減しました。ただし、フルプレシジョンに近い精度を確保するためには、各バイナリ化レベルに異なるフルプレシジョンスケーリング係数を導入する必要があり、計算集約型のMAC操作が必要です[6]。
-

The deployment of conventional ternarized neural
network (TNN) models to data sets such as electrocardiography (ECG) and electromyography (EMG) leads
to remarkable accuracy loss. In this article, we surpass
our prior works by quantizing both weights and activations of LSTMs into 2-bit ternary states with remarkable accuracy improvement compared to previous
work. We focus on reducing memory requirements
and computation costs through further reduction
of data bit-widths and hence, make the proposed
architectures extremely suitable for real-time and
resource-limited embedded and wearable devices. 
---
従来の三値化ニューラルネットワーク（TNN）モデルを心電図（ECG）や筋電図（EMG）などのデータセットに適用すると、著しい精度低下が生じます。本記事では、以前の研究と比較して、LSTMの重みと活性化の両方を2ビットの三値状態に量子化することで、精度を大幅に改善し、前回の研究を上回ります。データのビット幅をさらに削減することで、メモリ要件と計算コストを削減することに焦点を当て、提案されたアーキテクチャをリアルタイムでリソースが限られた組み込みおよびウェアラブルデバイスに非常に適したものにします。