# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。

Matrix multiplication (MatMul) typically dominates the overall computational
cost of large language models (LLMs). This cost only grows as LLMs scale to
larger embedding dimensions and context lengths. In this work, we show that
MatMul operations can be completely eliminated from LLMs while maintaining
strong performance at billion-parameter scales. Our experiments show that our
proposed MatMul-free models achieve performance on-par with state-of-the-art
Transformers that require far more memory during inference at a scale up to at least
2.7B parameters. We investigate the scaling laws and find that the performance
gap between our MatMul-free models and full precision Transformers narrows
as the model size increases. We also provide a GPU-efficient implementation
of this model which reduces memory usage by up to 61% over an unoptimized
baseline during training. By utilizing an optimized kernel during inference, our
model’s memory consumption can be reduced by more than 10× compared to
unoptimized models. To properly quantify the efficiency of our architecture,
we build a custom hardware solution on an FPGA which exploits lightweight
operations beyond what GPUs are capable of. We processed billion-parameter
scale models at 13W beyond human readable throughput, moving LLMs closer
to brain-like efficiency. This work not only shows how far LLMs can be stripped
back while still performing effectively, but also points at the types of operations
future accelerators should be optimized for in processing the next generation of
lightweight LLMs. 
---
この研究は、LLMが効果的に機能し続ける一方で、どれだけ簡素化できるかを示すだけでなく、次世代の軽量LLMを処理するために将来のアクセラレータが最適化されるべき操作の種類を指摘しています。
行列乗算（MatMul）は、通常、大規模言語モデル（LLM）の全体的な計算コストを支配しています。このコストは、LLMがより大きな埋め込み次元とコンテキスト長にスケールするにつれて増加します。本研究では、MatMul操作をLLMから完全に排除しながら、10億パラメータスケールで強力なパフォーマンスを維持できることを示します。実験では、提案するMatMulフリーモデルが、推論中にはるかに多くのメモリを必要とする最先端のトランスフォーマーと同等のパフォーマンスを達成することがわかりました。
スケーリング法則を調査した結果、MatMulフリーモデルと全精度トランスフォーマーの間のパフォーマンスギャップは、モデルサイズが大きくなるにつれて狭まることがわかりました。また、このモデルのGPU効率的な実装を提供し、トレーニング中に最適化されていないベースラインと比較して最大61％のメモリ使用量削減を実現しました。推論中に最適化されたカーネルを利用することで、モデルのメモリ消費は最適化されていないモデルと比較して10倍以上削減できます。アーキテクチャの効率を適切に定量化するために、FPGA上にカスタムハードウェアソリューションを構築し、GPUが可能な範囲を超える軽量操作を活用しました。人間が読み取れるスループットを超えて13Wで10億パラメータスケールのモデルを処理し、LLMを脳のような効率に近づけました。
この研究は、LLMが効果的に機能し続ける一方で、どれだけ簡素化できるかを示すだけでなく、次世代の軽量LLMを処理するために将来のアクセラレータが最適化されるべき操作の種類を指摘しています。

--
要約：
この研究では、行列乗算（MatMul）を完全に排除した大規模言語モデル（LLM）の開発を行い、10億パラメータスケールでの強力なパフォーマンスを維持しつつ、メモリ使用量を大幅に削減することに成功しました。提案されたモデルは、最先端のトランスフォーマーと同等の性能を発揮し、将来の軽量LLM向けのアクセラレータ最適化の方向性を示唆しています。

-
Matrix Multiplication (MatMul) is the dominant operation in most neural networks, where dense
layers involve vector-matrix multiplication (VMM), convolutions can be implemented as block-sparse
VMMs with shared weights, and self-attention relies on matrix-matrix multiplication (MMM). The
prevalence of MatMul is primarily due to Graphics Processing Units (GPUs) being optimized for
MatMul operations. By leveraging Compute Unified Device Architecture (CUDA) and highly optimized linear algebra libraries such as cuBLAS, the MatMul operation can be efficiently parallelized
and accelerated. This optimization was a key factor in the victory of AlexNet in the ILSVRC2012
competition and a historic marker for the rise of deep learning [1]. AlexNet notably utilized GPUs
to boost training speed beyond CPU capabilities, and as such, deep learning won the ‘hardware
lottery’ [2]. It also helped that both training and inference rely on MatMul.
---

MatMul（行列乗算）は、ほとんどのニューラルネットワークにおいて支配的な操作であり、密な層ではベクトル-行列乗算（VMM）が関与し、畳み込みは共有重みを持つブロックスパースVMMとして実装でき、自己注意は行列-行列乗算（MMM）に依存しています。MatMulの普及は主に、グラフィックス処理ユニット（GPU）がMatMul操作に最適化されているためです。Compute Unified Device Architecture（CUDA）とcuBLASなどの高度に最適化された線形代数ライブラリを活用することで、MatMul操作は効率的に並列化および加速できます。この最適化は、ILSVRC2012コンペティションでのAlexNetの勝利の重要な要因であり、深層学習の台頭の歴史的なマーカーでした[1]。AlexNetは特にGPUを利用してトレーニング速度をCPUの能力を超えて向上させたため、深層学習は「ハードウェアロッタリー」に勝利しました[2]。さらに、トレーニングと推論の両方がMatMulに依存していることも助けとなりました。
--

Despite its prevalence in deep learning, MatMul operations account for the dominant portion of
computational expense, often consuming the majority of the execution time and memory access during both training and inference phases. Several works have replaced MatMul with simpler operations
through two main strategies. The first strategy involves substituting MatMul with elementary
operations, e.g., AdderNet replaces multiplication with signed addition in convolutional neural
networks (CNNs) [3]. Given the focus on convolutions, AdderNet is intended for use in computer
vision over language modeling.
---
深層学習における普及にもかかわらず、MatMul操作は計算コストの大部分を占め、トレーニングと推論の両方のフェーズで実行時間とメモリアクセスの大部分を消費することがよくあります。いくつかの研究では、主に2つの戦略を通じてMatMulをより単純な操作に置き換えています。最初の戦略は、MatMulを基本的な操作に置き換えることです。たとえば、AdderNetは畳み込みニューラルネットワーク（CNN）で乗算を符号付き加算に置き換えます[3]。畳み込みに焦点を当てているため、AdderNetは言語モデリングよりもコンピュータビジョンでの使用を意図しています。

--
The second approach employs binary or ternary quantization, simplifying MatMul to operations
where values are either flipped or zeroed out before accumulation. Quantization can be applied to
either activations or weights: spiking neural networks (SNNs) use binarized activations [4, 5, 6],
while binary and ternary neural networks (BNNs and TNNs) use quantized weights [7]. Both methods
can also be combined [8, 9].
---
2番目のアプローチは、バイナリまたは三元量子化を使用して、MatMulを累積前に値が反転またはゼロ化される操作に簡素化します。量子化は、アクティベーションまたは重みに適用できます。スパイキングニューラルネットワーク（SNN）はバイナリ化されたアクティベーションを使用し[4, 5, 6]、バイナリニューラルネットワーク（BNN）および三元ニューラルネットワーク（TNN）は量子化された重みを使用します[7]。両方の方法を組み合わせることもできます[8, 9]。

--
Recent advances in language modeling, like BitNet [10, 11], demonstrate quantization’s scalability,
replacing all dense layer weights with binary/ternary values to support up to 3 billion parameters.
Despite replacing VMMs with accumulations in all dense layers, BitNet retains the self-attention
mechanism which relies on an expensive MMM. Dynamically computed matrices Q (query) and
K (key) are multiplied to form the attention map. Since both Q and K matrices are dynamically
computed from pre-activation values, achieving optimal hardware efficiency on GPUs requires
custom optimizations, such as specialized kernels and advanced memory access patterns. Despite
these efforts, such MatMul operations remain resource-intensive on GPUs, as they involve extensive
data movement and synchronization which can significantly hinder computational throughput and
efficiency [12]. In our experiments, ternary quantization of the attention matrices in BitNet causes a
significant drop in performance and failure to reach model convergence (see Fig. 1). This raises the
question: is it possible to completely eliminate MatMul from LLMs?

---
言語モデリングの最近の進歩、例えばBitNet [10, 11]は、量子化のスケーラビリティを示しており、すべての密な層の重みをバイナリ/三元値に置き換えて最大30億パラメータをサポートしています。BitNetはすべての密な層でVMMを累積に置き換えていますが、高価なMMMに依存する自己注意メカニズムを保持しています。動的に計算される行列Q（クエリ）とK（キー）が掛け合わされて注意マップが形成されます。QとKの両方の行列は前活性化値から動的に計算されるため、GPU上で最適なハードウェア効率を達成するには、専門化されたカーネルや高度なメモリアクセスパターンなどのカスタム最適化が必要です。これらの努力にもかかわらず、そのようなMatMul操作は依然としてGPU上でリソース集約的であり、大量のデータ移動と同期を伴うため、計算スループットと効率を大幅に妨げる可能性があります[12]。私たちの実験では、BitNetの注意行列の三元量子化がパフォーマンスに大きな低下を引き起こし、モデル収束に失敗します（図1を参照）。これは、LLMからMatMulを完全に排除することが可能かどうかという疑問を提起します。

--
In this work, we develop the first scalable MatMul-free language model (Matmul-free LM) by using
additive operations in dense layers and element-wise Hadamard products for self-attention-like
functions. Specifically, ternary weights eliminate MatMul in dense layers, similar to BNNs. To
remove MatMul from self-attention, we optimize the Gated Recurrent Unit (GRU) [13] to rely solely
on element-wise products and show that this model competes with state-of-the-art Transformers
while eliminating all MatMul operations.
---
本研究では、密な層で加算操作を使用し、自己注意のような関数に要素ごとのHadamard積を使用することで、スケーラブルなMatMulフリー言語モデル（Matmul-free LM）を初めて開発しました。具体的には、三元重みを使用して密な層でのMatMulを排除し、BNNと同様にします。自己注意からMatMulを排除するために、Gated Recurrent Unit（GRU）[13]を最適化し、要素ごとの積のみに依存することを示し、このモデルが最先端のトランスフォーマーと競合しながらすべてのMatMul操作を排除することを示します。

--
To quantify the hardware benefits of lightweight models, we provide an optimized GPU implementation in addition to a custom FPGA accelerator. By using fused kernels in the GPU implementation of
the ternary dense layers, training is accelerated by 25.6% and memory consumption is reduced by
up to 61.0% over an unoptimized baseline on GPU. Furthermore, by employing lower-bit optimized
CUDA kernels, inference speed is increased by 4.57 times, and memory usage is reduced by a
factor of 10 when the model is scaled up to 13B parameters. This work goes beyond software-only
implementations of lightweight models and shows how scalable, yet lightweight, language models
can both reduce computational demands and energy use in the real-world.
---
軽量モデルのハードウェアの利点を定量化するために、カスタムFPGAアクセラレータに加えて、最適化されたGPU実装を提供します。三元密な層のGPU実装で融合カーネルを使用することで、トレーニングは25.6％加速され、最適化されていないベースラインと比較してメモリ消費が最大61.0％削減されます。さらに、低ビットの最適化されたCUDAカーネルを使用することで、推論速度は4.57倍向上し、モデルが13Bパラメータにスケールアップされるとメモリ使用量は10倍に削減されます。この研究は、軽量モデルのソフトウェアのみの実装を超え、スケーラブルでありながら軽量な言語モデルが計算要求とエネルギー使用を現実世界でどのように削減できるかを示しています。

--
Binary, Ternary, and Low-Precision Quantization for Language Models: The effort to quantize
language models began with reducing a ternary BERT into a binarized model [14], achieving 41%
average accuracy on the GLUE benchmarks with subsequent fine-tuning. Ref. [15] distilled the
intermediate outputs from a full precision BERT to a quantized version. Recently, Ref. [16] introduced
an incremental quantization approach, progressively quantizing a model from 32-bit to 4-bit, 2-bit,
and finally to binary model parameters. Following the quantization of BERT, low-precision language
generation models have gained momentum. Ref. [17] used Quantization-Aware Training (QAT) to
successfully train a model with 2-bit weights. BitNet pushed this to 3-billion-parameter binary and
ternary models while maintaining competitive performance with Llama-like language models [10, 11].
---
言語モデルのバイナリ、三元、および低精度量子化：言語モデルの量子化の取り組みは、三元BERTをバイナリ化されたモデルに縮小することから始まり、GLUEベンチマークで41％の平均精度を達成しました[14]。Ref. [15]は、全精度BERTから中間出力を蒸留して量子化されたバージョンに変換しました。最近、Ref. [16]は、32ビットから4ビット、2ビット、そして最終的にバイナリモデルパラメータに段階的に量子化する増分量子化アプローチを導入しました。BERTの量子化に続いて、低精度の言語生成モデルが勢いを増しています。Ref. [17]は、量子化認識トレーニング（QAT）を使用して2ビット重みでモデルを成功裏にトレーニングしました。BitNetはこれを3億パラメータのバイナリおよび三元モデルに押し上げ、Llamaのような言語モデルと競争力のあるパフォーマンスを維持しました[10, 11]。

--
MatMul-free Transformers: The use of MatMul-free Transformers has been largely concentrated
in the domain of SNNs. Spikformer led the first integration of the Transformer architecture with
SNNs [18, 19], with later work developing alternative Spike-driven Transformers [20, 21]. These
techniques demonstrated success in vision tasks. In the language understanding domain, SpikingBERT [22] and SpikeBERT [23] applied SNNs to BERT utilizing knowledge distillation techniques to
perform sentiment analysis. In language generation, SpikeGPT trained a 216M-parameter generative
model using a spiking RWKV architecture. However, these models remain constrained in size, with SpikeGPT being the largest, reflecting the challenges of scaling with binarized activations. In addition
to SNNs, BNNs have also made significant progress in this area. BinaryViT [24] and BiViT [25]
successfully applied Binary Vision Transformers to visual tasks. Beyond these approaches, Kosson et
al. [26] achieve multiplication-free training by replacing multiplications, divisions, and non-linearities
with piecewise affine approximations while maintaining performance.
---
MatMulフリートランスフォーマー：MatMulフリートランスフォーマーの使用は、主にSNNの領域に集中しています。Spikformerは、TransformerアーキテクチャとSNNの最初の統合を導入し[18, 19]、後の研究では代替のスパイク駆動トランスフォーマーを開発しました[20, 21]。これらの技術は、視覚タスクで成功を収めました。言語理解ドメインでは、SpikingBERT [22]とSpikeBERT [23]がSNNをBERTに適用し、知識蒸留技術を利用して感情分析を行いました。言語生成では、SpikeGPTがスパイキングRWKVアーキテクチャを使用して216Mパラメータの生成モデルをトレーニングしました。しかし、これらのモデルはサイズに制約があり、SpikeGPTが最大であり、バイナリ化されたアクティベーションでのスケーリングの課題を反映しています。SNNに加えて、BNNもこの分野で大きな進展を遂げています。BinaryViT [24]とBiViT [25]は、バイナリビジョントランスフォーマーを視覚タスクに成功裏に適用しました。これらのアプローチを超えて、Kosson et al. [26]は乗算、除算、および非線形性を区分線形近似に置き換えることで乗算なしのトレーニングを実現しながらパフォーマンスを維持しています。

--
In this section, we break down the components of the proposed MatMul-free LM. We first describe
the MatMul-free dense layers (BitLinear layers) that use ternary weights. By constraining the
weights to the set {−1, 0, +1} and applying additional quantization techniques, MatMul operations
are replaced with addition and negation operations. This reduces computational cost and memory
utilization, while preserving the expressiveness of the network. We then provide further detail of our
MatMul-free LM architecture, which includes a token mixer for capturing sequential dependencies
and a channel mixer for integrating information across embedding dimensions.
---
このセクションでは、提案されたMatMulフリーLMのコンポーネントを分解します。最初に、三元重みを使用するMatMulフリーデンス層（BitLinear層）について説明します。重みをセット{−1、0、+1}に制約し、追加の量子化技術を適用することで、MatMul操作は加算および否定操作に置き換えられます。これにより、計算コストとメモリ使用量が削減されながら、ネットワークの表現力が保持されます。その後、シーケンシャルな依存関係を捉えるためのトークンミキサーと、埋め込み次元全体で情報を統合するためのチャネルミキサーを含むMatMulフリーLMアーキテクチャの詳細を提供します。
--

The Method section is structured as follows. First, in Sec. 3.1, we provide a comprehensive description
of the MatMul-free dense layers with ternary weights, which form the foundation of our approach.
Next, Sec. 3.2 introduces our hardware-efficient fused BitLinear layer, designed to optimize the
implementation of BitLinear layers. Building upon these components, Sec. 3.3 delves into the details
of our MatMul-free LM architecture. We present the MatMul-free token mixer, where we propose
the MatMul-free Linear Gated Recurrent Unit (MLGRU), and the MatMul-free channel mixer, which
employs the Gated Linear Unit (GLU) with BitLinear layers. By combining the MLGRU token mixer
and the GLU channel mixer with ternary weights, our proposed architecture relies solely on addition
and element-wise products. Finally, Sec. 3.4 provides an overview of the training details used to
optimize our model.
---
メソッドセクションは次のように構成されています。まず、Sec. 3.1では、アプローチの基礎を形成する三元重みを持つMatMulフリーデンス層の包括的な説明を提供します。次に、Sec. 3.2では、BitLinear層の実装を最適化するために設計されたハードウェア効率的な融合BitLinear層を紹介します。これらのコンポーネントに基づいて、Sec. 3.3では、MatMulフリーLMアーキテクチャの詳細に掘り下げます。MatMulフリートークンミキサーを提示し、MatMulフリー線形ゲート付きリカレントユニット（MLGRU）を提案し、BitLinear層を使用したゲート付き線形ユニット（GLU）を採用したMatMulフリーチャネルミキサーを紹介します。MLGRUトークンミキサーとGLUチャネルミキサーを三元重みと組み合わせることで、提案されたアーキテクチャは加算と要素ごとの積のみに依存します。最後に、Sec. 3.4では、モデルを最適化するために使用されるトレーニングの詳細について概説します。

--
To avoid using standard MatMul-based dense layers, we adopt
BitNet to replace dense layers containing MatMuls with BitLinear modules, which use ternary
weights to transform MatMul operations into pure addition operation with accumulation, i.e., ternary
accumulation. When using ternary weights, the elements from the weight matrix W are constrained
to values from the set {−1, 0, +1}. Let Wf denote the ternary weight matrix.
---
MatMulベースの密な層を使用しないために、BitNetを採用して、MatMulを含む密な層をBitLinearモジュールに置き換えます。これらのモジュールは三元重みを使用して、MatMul操作を純粋な加算操作に変換します。つまり、三元累積です。三元重みを使用する場合、重み行列Wの要素はセット{−1、0、+1}からの値に制約されます。Wfを三元重み行列とします。

-
We adopt the perspective from Metaformer [27], which suggests that Transformers consist of a tokenmixer (for mixing temporal information, i.e., Self Attention [28], Mamba [29]) and a channel-mixer
(for mixing embedding/spatial information, i.e., feed-forward network, Gated Linear Unit (GLU)
[30, 31]). A high-level overview of the architecture is shown in Fig. 2.
---
Metaformer [27]の視点を採用し、トランスフォーマーはトークンミキサー（時間情報を混合するため、つまり自己注意[28]、マンバ[29]）とチャネルミキサー（埋め込み/空間情報を混合するため、つまりフィードフォワードネットワーク、ゲート付き線形ユニット（GLU）[30, 31]）で構成されることを示唆しています。アーキテクチャの高レベルの概要を図2に示します。
--

Self-attention is the most common token mixer in modern language models, relying on matrix
multiplication between three matrices: Q, K, and V . To convert these operations into additions, we
binarize or ternarize at least two of the matrices. Assuming all dense layer weights are ternary, we
quantize Q and K, resulting in a ternary attention map that eliminates multiplications in self-attention.
However, as shown in Fig. 1, the model trained this way fails to converge. One possible explanation
is that activations contain outliers crucial for performance but difficult to quantize effectively [32, 33].
To address this challenge, we explore alternative methods for mixing tokens without relying on matrix
multiplications
---
自己注意は、現代の言語モデルで最も一般的なトークンミキサーであり、3つの行列Q、K、およびVの間の行列乗算に依存しています。これらの操作を加算に変換するためには、少なくとも2つの行列をバイナリ化または三元化します。すべての密な層の重みが三元であると仮定すると、QとKを量子化し、自己注意における乗算を排除する三元注意マップを生成します。しかし、図1に示すように、この方法でトレーニングされたモデルは収束に失敗します。可能な説明の1つは、アクティベーションにパフォーマンスに重要だが効果的に量子化するのが難しい外れ値が含まれていることです[32, 33]。この課題に対処するために、行列乗算に依存せずにトークンを混合する代替方法を探ります。

-

By resorting to the use of ternary RNNs, which combine element-wise operations and accumulation,
it becomes possible to construct a MatMul-free token mixer. Among various RNN architectures, the
GRU is noted for its simplicity and efficiency, achieving similar performance to Long Short-Term
Memory (LSTM) [34] cells while using fewer gates and having a simpler structure. Thus, we choose
the GRU as the foundation for building a MatMul-free token mixer. We first revisit the standard GRU
and then demonstrate, step by step, how we derive the MLGRU
---
三元RNNを使用することで、要素ごとの操作と累積を組み合わせることができ、MatMulフリートークンミキサーを構築することが可能になります。さまざまなRNNアーキテクチャの中で、GRUはそのシンプルさと効率性で知られており、Long Short-Term Memory（LSTM）[34]セルと同様のパフォーマンスを達成しながら、より少ないゲートを使用し、構造が簡単です。したがって、GRUをMatMulフリートークンミキサーを構築するための基礎として選択します。まず、標準のGRUを再検討し、MLGRUを導出する方法を段階的に示します。

-
A key characteristic of the GRU is the coupling of the input gate vector ft
and the forget gate vector
(1 − ft
), which together constitute the ‘leakage’ unit. This leakage unit decays the hidden state ht−1
and the candidate hidden state ct through element-wise multiplication, see Eq. 4. This operation
allows the model to adaptively retain information from the previous hidden state ht−1 and incorporate
new information from the candidate hidden state ct. Importantly, this operation relies solely on
element-wise multiplication, avoiding the need for the MatMul. We aim to preserve this property of
the GRU while introducing further modifications to create a MatMul-free variant of the model.
---
GRUの重要な特性は、入力ゲートベクトルftと忘却ゲートベクトル(1 − ft)の結合であり、これらは一緒に「漏れ」ユニットを構成します。この漏れユニットは、要素ごとの乗算を通じて隠れ状態ht−1と候補隠れ状態ctを減衰させます（式4を参照）。この操作により、モデルは前の隠れ状態ht−1からの情報を適応的に保持し、候補隠れ状態ctからの新しい情報を組み込むことができます。重要なのは、この操作が要素ごとの乗算のみに依存し、MatMulの必要性を回避することです。モデルのMatMulフリー変種を作成するために、GRUのこの特性を保持しつつ、さらに変更を加えることを目指します。

-

We first remove hidden-state related weights Wcc,
Whr, Whf , and the activation between hidden states (tanh). This modification not only makes the
model MatMul-free but also enables parallel computation similar to Transformers. This approach is
critical for improving computational efficiency, as transcendental functions are expensive to compute
accurately, and non-diagonal matrices in the hidden-state would hinder parallel computations. This
modification is a key feature of recent RNNs, such as the Linear Recurrent Unit [35], Hawk [36], and
RWKV-4 [37]. We then add a data-dependent output gate between ht and ot, inspired by the LSTM
and widely adopted by recent RNN models

---
最初に、隠れ状態関連の重みWcc、Whr、Whf、および隠れ状態間の活性化（tanh）を削除します。この変更により、モデルはMatMulフリーになり、トランスフォーマーと同様の並列計算が可能になります。このアプローチは、超越関数の正確な計算が高価であり、隠れ状態の非対角行列が並列計算を妨げるため、計算効率を改善するために重要です。この変更は、Linear Recurrent Unit [35]、Hawk [36]、およびRWKV-4 [37]などの最近のRNNの重要な特徴です。次に、LSTMからインスパイアを受けたデータ依存型出力ゲートをhtとotの間に追加し、最近のRNNモデルで広く採用されています。

-
Following the approach of HGRN [38], we further simplify the computation of the candidate hidden
state by keeping it as a simple linear transform, rather than coupling it with the hidden state. This can
be rewritten as a linear transformation of the input. Finally, we replace all remaining weight matrices
with ternary weight matrices, completely removing MatMul operations. The resulting MLGRU
architecture can be formalized as follows
---
HGRN [38]のアプローチに従い、候補隠れ状態の計算を隠れ状態と結合するのではなく、単純な線形変換として保持することでさらに簡素化します。これは、入力の線形変換として書き直すことができます。最後に、残りのすべての重み行列を三元重み行列に置き換え、MatMul操作を完全に排除します。結果として得られるMLGRUアーキテクチャは次のように形式化できます。

-
 The MLGRU
can be viewed as a simplified variant of HGRN that omits complex-valued components and reduces
the hidden state dimension from 2d to d. This simplification makes MLGRU more computationally
efficient while preserving essential gating mechanisms and ternary weight quantization.
---
MLGRUは、複素数成分を省略し、隠れ状態の次元を2dからdに削減したHGRNの簡略化されたバリアントと見なすことができます。この簡素化により、MLGRUは計算効率が向上し、重要なゲーティングメカニズムと三元重み量子化を保持します。

-
Alternatively to the MLGRU, which employs a data-dependent decay with element-wise product
hidden state, the a similarly modified version of the RWKV-4 model can also satisfy the requirement
of a MatMul-free token mixer, utilizing static decay and normalization. The performance of using
RWKV-4 as a MatMul-free token mixer is discussed in the Experiment section, with a detailed description of the RWKV-4 model provided in Appendix B. However, RWKV-4 introduces exponential
and division operations, which are less hardware-efficient compared to the MLGRU.
---
MLGRUを使用する代わりに、要素ごとの積隠れ状態を持つデータ依存型減衰を採用するRWKV-4モデルの同様に変更されたバージョンも、MatMulフリートークンミキサーの要件を満たすことができます。RWKV-4をMatMulフリートークンミキサーとして使用するパフォーマンスについては、実験セクションで議論され、RWKV-4モデルの詳細な説明は付録Bに提供されています。ただし、RWKV-4は指数関数と除算操作を導入し、MLGRUと比較してハードウェア効率が低くなります。

-
The GLU consists of three main steps: 1) upscaling the t-step input xt to gt,ut using weight matrices Wg,Wu 2) elementwise gating ut with gt followed by a nonlinearity f(·), where we apply Swish [31]. 3) Down-scaling the gated representation pt back to the original size through a linear transformation Wd. Following Llama [39], we maintain the overall number of parameters of GLU at 8d^2 by setting the upscaling factor to 8d/3.
---
GLUは3つの主要なステップで構成されています：1）重み行列Wg,Wuを使用してtステップ入力xtをgt,utにアップスケーリングする。2）gtでutを要素ごとにゲーティングし、Swish [31]を適用する非線形性f(·)を続ける。3）ゲートされた表現ptを線形変換Wdを通じて元のサイズにダウンスケーリングします。Llama [39]に従い、アップスケーリング係数を8d/3に設定することで、GLUの全体的なパラメータ数を8d^2に維持します。
-

The channel mixer here only consists of dense layers, which are replaced with ternary accumulation
operations. By using ternary weights in the BitLinear modules, we can eliminate the need for
expensive MatMuls, making the channel mixer more computationally efficient while maintaining its
effectiveness in mixing information across channels.
---
ここでのチャネルミキサーは密な層のみで構成され、三元累積操作に置き換えられます。BitLinearモジュールで三元重みを使用することで、高価なMatMulの必要性を排除し、チャネルミキサーをより計算効率的にしながら、チャネル間の情報を混合する効果を維持します。
-

To handle non-differentiable functions such as the Sign and Clip functions
during backpropagation, we use the straight-through estimator (STE) [43] as a surrogate function for
the gradient. STE allows gradients to flow through the network unaffected by these non-differentiable
functions, enabling the training of our quantized model. This technique is widely adopted in BNNs
and SNNs.

---
非微分可能な関数（Sign関数やClip関数など）を逆伝播中に処理するために、直通推定器（STE）[43]を勾配の代理関数として使用します。STEは、これらの非微分可能な関数の影響を受けずにネットワークを通じて勾配が流れることを可能にし、量子化されたモデルのトレーニングを可能にします。この技術はBNNやSNNで広く採用されています。\

-
When training a language model with ternary weights, using the same
learning rate as regular models can lead to excessively small updates that have no impact on the
clipping operation. This prevents weights from being effectively updated and results in biased
gradients and update estimates based on the ternary weights. To address this challenge, it is common
practice to employ a larger learning rate when training binary or ternary weight language models, as it
facilitates faster convergence [44, 45, 11]. In our experiments, we maintain consistent learning rates
across both the 370M and 1.3B models, aligning with the approach described in Ref. [46]. Specifically,
for the Transformer++ model, we use a learning rate of 3e − 4, while for the MatMul-free LM, we
employ a learning rate of 4e − 3, 2.5e − 3, 1.5e − 3 in 370M, 1.5B and 2.7B, respectively. These
learning rates are chosen based on the most effective hyperparameter sweeps for faster convergence
during the training process.
---
言語モデルを三元重みでトレーニングする際、通常のモデルと同じ学習率を使用すると、クリッピング操作に影響を与えないほど小さな更新が行われる可能性があります。これにより、重みが効果的に更新されず、三元重みに基づくバイアスのかかった勾配と更新推定が生じます。この課題に対処するために、バイナリまたは三元重みの言語モデルをトレーニングする際には、より大きな学習率を使用することが一般的な慣行です。これにより、より速い収束が促進されます[44, 45, 11]。私たちの実験では、370Mモデルと1.3Bモデルの両方で一貫した学習率を維持し、Ref. [46]で説明されたアプローチに合わせています。具体的には、Transformer++モデルでは学習率3e − 4を使用し、MatMulフリーLMでは370M、1.5B、および2.7Bでそれぞれ4e − 3、2.5e − 3、1.5e − 3の学習率を採用しています。これらの学習率は、トレーニングプロセス中のより速い収束のための最も効果的なハイパーパラメータスイープに基づいて選択されています。
-

When training conventional Transformers, it is common practice to
employ a cosine learning rate scheduler and set a minimal learning rate, typically 0.1× the initial
learning rate. We follow this approach when training the full precision Transformer++ model. However, for the MatMul-free LM, the learning dynamics differ from those of conventional Transformer
language models, necessitating a different learning strategy. We begin by maintaining the cosine
learning rate scheduler and then reduce the learning rate by half midway through the training process.
---
従来のトランスフォーマーをトレーニングする際、コサイン学習率スケジューラを使用し、通常は初期学習率の0.1倍の最小学習率を設定することが一般的な慣行です。フル精度のTransformer++モデルをトレーニングする際には、このアプローチに従います。ただし、MatMulフリーLMでは、学習ダイナミクスが従来のトランスフォーマー言語モデルとは異なるため、異なる学習戦略が必要です。最初はコサイン学習率スケジューラを維持し、トレーニングプロセスの途中で学習率を半分に減少させます。
-

The RTL implementation of the MatMul-free token generation core is deployed on a D5005 Stratix
10 programmable acceleration card (PAC) in the Intel FPGA Devcloud. The core completes a
forward-pass of a block in 43ms at d = 512 and achieves a clock rate of 60MHz. The resource
utilization, power and performance of the single-core implementation of a single block (N = 1) are
shown in Tab. 2. ‘% ALM Core’ refers to the percentage of the total adaptive logic modules used by
the core logic, and ‘%ALM Total’ includes the core, the additional interconnect/arbitration logic, and
“shell” logic for the FPGA Interface Manager. ‘M20K’ refers to the utilization of the memory blocks,
and indicates that the number of cores are constrained by ALMs, and not on-chip memory (for this
DDR implementation). We implement a single token generation core, and estimate the total number
of cores that could fit on the platform and the corresponding power, performance and area impact.
This is the simplest case where the core only receives 8 bits at a time from memory.
---
RTL実装のMatMulフリートークン生成コアは、Intel FPGA DevcloudのD5005 Stratix 10プログラム可能アクセラレーションカード（PAC）に展開されています。このコアは、d = 512でブロックのフォワードパスを43msで完了し、60MHzのクロックレートを達成します。単一ブロック（N = 1）の単一コア実装のリソース使用率、電力、およびパフォーマンスを表2に示します。「% ALM Core」は、コアロジックで使用される総適応ロジックモジュールの割合を指し、「%ALM Total」にはコア、追加の相互接続/仲裁ロジック、およびFPGAインターフェイスマネージャー用の「シェル」ロジックが含まれます。「M20K」はメモリブロックの使用率を指し、コアの数がALMによって制約されており、オンチップメモリ（このDDR実装の場合）ではないことを示しています。単一のトークン生成コアを実装し、プラットフォームに適合する可能性のあるコアの総数と対応する電力、パフォーマンス、および面積への影響を推定します。これは、コアがメモリから一度に8ビットのみを受信する最も単純なケースです。
-

We have demonstrated the feasibility and effectiveness of the first scalable MatMul-free language
model. Our work challenges the paradigm that MatMul operations are indispensable for building
high-performing language models and paves the way for the development of more efficient and
hardware-friendly architectures. We achieve performance on par with state-of-the-art Transformers
while eliminating the need for MatMul operations, with an optimized implementation that significantly
enhances both training and inference efficiency, reducing both memory usage and latency. As the
demand for deploying language models on various platforms grows, MatMul-free LMs present a
promising direction for creating models that are both effective and resource-efficient. However,
one limitation of our work is that the MatMul-free LM has not been tested on extremely large-scale
models (e.g., 100B+ parameters) due to computational constraints. This work serves as a call to
action for institutions and organizations that have the resources to build the largest language models
to invest in accelerating lightweight models. By prioritizing the development and deployment of
MatMul-free architectures such as this one, the future of LLMs will only become more accessible,
efficient, and sustainable.
---
私たちは、最初のスケーラブルなMatMulフリー言語モデルの実現可能性と効果を実証しました。この研究は、MatMul操作が高性能な言語モデルを構築するために不可欠であるというパラダイムに挑戦し、より効率的でハードウェアに優しいアーキテクチャの開発への道を開きます。MatMul操作を排除しながら、最先端のトランスフォーマーと同等のパフォーマンスを達成し、トレーニングと推論の効率を大幅に向上させる最適化された実装により、メモリ使用量とレイテンシを削減します。さまざまなプラットフォームで言語モデルを展開する需要が高まる中で、MatMulフリーLMは効果的でリソース効率の高いモデルを作成するための有望な方向性を示しています。ただし、計算制約により、MatMulフリーLMは非常に大規模なモデル（例：100B以上のパラメータ）ではテストされていないという制限があります。この研究は、最大の言語モデルを構築するリソースを持つ機関や組織に対して、軽量モデルの加速に投資するよう呼びかけるものです。このようなMatMulフリーアーキテクチャの開発と展開を優先することで、LLMの未来はよりアクセスしやすく、効率的で持続可能になるでしょう。
-

: Performance comparison and analysis of different models and configurations. (a) and (b)
show the training performance comparison between Vanilla BitLinear and Fused BitLinear in terms
of time and memory consumption as a function of batch size. (c) illustrates the effect of learning
rate on training loss for the MatMul-free LM. (d) compares the inference memory consumption and
latency between MatMul-free LM and Transformer++ across various model sizes.
---
パフォーマンスの比較と分析：異なるモデルと構成のパフォーマンス比較と分析。(a)および(b)は、バッチサイズの関数としての時間とメモリ消費に関するVanilla BitLinearとFused BitLinearのトレーニングパフォーマンス比較を示しています。(c)は、MatMulフリーLMのトレーニング損失に対する学習率の効果を示しています。(d)は、さまざまなモデルサイズにわたるMatMulフリーLMとTransformer++の推論メモリ消費とレイテンシを比較しています。
-
 Zero-shot accuracy of MatMul-free LM and Transformer++ on benchmark datasets.
---
MatMulフリーLMとTransformer++のベンチマークデータセットにおけるゼロショット精度。
-
Our primary focus is testing the MatMul-free LM on moderate-scale language modeling tasks.
We compare two variants of our MatMul-free LM against a reproduced advanced Transformer
architecture (Transformer++, based on Llama-2) across three model sizes: 370M, 1.3B, and 2.7B
parameters. For a fair comparison, all models are pre-trained on the SlimPajama dataset [47], with
the 370M model trained on 15 billion tokens, and the 1.3B and 2.7B models trained on 100 billion
tokens each. All experiments were conducted using the flash-linear-attention [48] framework, with
the Mistral [42] tokenizer (vocab size: 32k) and optimized triton kernel. The training of our models
was conducted using 8 NVIDIA H100 GPUs. The training duration was approximately 5 hours for
the 370M model, 84 hours for the 1.3B model, and 173 hours for the 2.7B model.
---
私たちの主な焦点は、中規模の言語モデリングタスクでMatMulフリーLMをテストすることです。3つのモデルサイズ（370M、1.3B、および2.7Bパラメータ）にわたって、MatMulフリーLMの2つのバリアントを再現された高度なトランスフォーマーアーキテクチャ（Transformer++、Llama-2に基づく）と比較します。公正な比較のために、すべてのモデルはSlimPajamaデータセット[47]で事前トレーニングされ、370Mモデルは150億トークンでトレーニングされ、1.3Bおよび2.7Bモデルはそれぞれ1000億トークンでトレーニングされました。すべての実験は、flash-linear-attention [48]フレームワークを使用して実施され、Mistral [42]トークナイザー（語彙サイズ：32k）と最適化されたtritonカーネルを使用しました。モデルのトレーニングは8つのNVIDIA H100 GPUを使用して行われました。トレーニング期間は、370Mモデルで約5時間、1.3Bモデルで84時間、2.7Bモデルで173時間でした。
-

Neural scaling laws posit that model error decreases as a power function of training set size and model
size, and have given confidence in performance. Such projections become important as training
becomes increasingly expensive with larger models. A widely adopted best practice in LLM training
is to first test scalability with smaller models, where scaling laws begin to take effect [49, 50, 51].
The GPT-4 technical report revealed that a prediction model just 1/10, 000 the size of the final model
can still accurately forecast the full-sized model performance [52].
We evaluate how the scaling law fits to the 370M, 1.3B and 2.7B parameter models in both Transformer++ and MatMul-free LM, shown in Fig. 3. For a conservative comparison, each operation
is treated identically between MatMul-free LM and Transformer++. But note that all weights and
activations in Transformer++ are in BF16, while BitLinear layers in MatMul-free LM use ternary
parameters, with BF16 activations. As such, an average operation in MatMul-free LM will be
computationally cheaper than that of Transformer++.
Interestingly, the scaling projection for the MatMul-free LM exhibits a steeper descent compared
to that of Transformer++. This suggests that the MatMul-free LM is more efficient in leveraging additional compute resources to improve performance. As a result, the scaling curve of the
MatMul-free LM is projected to intersect with the scaling curve of Transformer++ at approximately 1023 FLOPs. This compute scale is roughly equivalent to the training FLOPs required for Llama-3
8B (trained with 15 trillion tokens) and Llama-2 70B (trained with 2 trillion tokens), suggesting that
MatMul-free LM not only outperforms in efficiency, but can also outperform in terms of loss when
scaled up.
---
ニューラルスケーリング法則は、モデルエラーがトレーニングセットサイズとモデルサイズのべき関数として減少すると仮定し、パフォーマンスに自信を与えています。このような予測は、トレーニングが大規模なモデルでますます高価になるにつれて重要になります。LLMトレーニングで広く採用されているベストプラクティスは、スケーリング法則が効果を発揮し始める小規模モデルで最初にスケーラビリティをテストすることです[49, 50, 51]。GPT-4技術レポートでは、最終モデルの1/10,000のサイズの予測モデルでも、フルサイズのモデルパフォーマンスを正確に予測できることが明らかになりました[52]。
370M、1.3B、および2.7Bパラメータモデルのスケーリング法則がTransformer++とMatMulフリーLMの両方にどのように適合するかを評価します。図3に示されています。保守的な比較のために、各操作はMatMulフリーLMとTransformer++の間で同一に扱われます。ただし、Transformer++のすべての重みとアクティベーションはBF16であり、MatMulフリーLMのBitLinear層は三元パラメータを使用し、BF16アクティベーションを持つことに注意してください。そのため、MatMulフリーLMの平均操作はTransformer++よりも計算コストが安くなります。
興味深いことに、MatMulフリーLMのスケーリング予測は、Transformer++のそれよりも急な下降を示しています。これは、MatMulフリーLMが追加の計算リソースを活用してパフォーマンスを向上させるのにより効率的であることを示唆しています。その結果、MatMulフリーLMのスケーリング曲線は、約1023 FLOPsでTransformer++のスケーリング曲線と交差することが予測されています。この計算スケールは、Llama-3 8B（15兆トークンでトレーニング）およびLlama-2 70B（2兆トークンでトレーニング）に必要なトレーニングFLOPsとほぼ同等であり、MatMulフリーLMは効率性だけでなく、スケールアップ時の損失においても優れている可能性があることを示唆しています。
-

Scaling law comparison between MatMul-free LM and Transformer++ models, depicted
through their loss curves. The red lines represent the loss trajectories of the MatMul-free LM, while
the blue lines indicate the losses of the Transformer++ models. The star marks the intersection point
of the scaling law projection for both model types. MatMul-free LM uses ternary parameters and
BF16 activations, whereas Transformer++ uses BF16 parameters and activations.
---
スケーリング法則の比較：MatMulフリーLMとTransformer++モデルの損失曲線を通じて描かれています。赤い線はMatMulフリーLMの損失軌道を表し、青い線はTransformer++モデルの損失を示しています。星印は、両方のモデルタイプのスケーリング法則予測の交差点を示します。MatMulフリーLMは三元パラメータとBF16アクティベーションを使用し、Transformer++はBF16パラメータとアクティベーションを使用しています。
-

Interestingly, we observed that during the final training stage, when the network’s learning rate
approaches 0, the loss decreases significantly, exhibiting an S-shaped loss curve. This phenomenon
has also been reported by [11, 44] when training binary/ternary language models.
---
興味深いことに、ネットワークの学習率が0に近づく最終トレーニング段階で、損失が大幅に減少し、S字型の損失曲線を示すことを観察しました。この現象は、バイナリ/三元言語モデルのトレーニング時に[11, 44]によっても報告されています。
-

In line with benchmarking in BitNet, we evaluated the zero-shot performance of these models
on a range of language tasks, including ARC-Easy [53], ARC-Challenge [53], Hellaswag [54],
Winogrande [55], PIQA [56], and OpenbookQA [57]. The results are shown in Tab. 1. Details about
the datasets can be found in Appendix C. All evaluations are performed using the LM evaluation
harness [58]. The MatMul-free LM models achieve competitive performance compared to the
Transformer++ baselines across all tasks, demonstrating its effectiveness in zero-shot learning despite
the absence of MatMul operations, and the lower memory required from ternary weights. Notably,
the 2.7B MatMul-free LM model outperforms its Transformer++ counterpart on ARC-Challenge
and OpenbookQA, while maintaining comparable performance on the other tasks. As the model
size increases, the performance gap between MatMul-free LM and Transformer++ narrows, which is
consistent with the scaling law. These results highlight that MatMul-free architectures are capable
achieving strong zero-shot performance on a diverse set of language tasks, ranging from question
answering and commonsense reasoning to physical understanding.
---
BitNetでのベンチマークに沿って、これらのモデルのゼロショットパフォーマンスを、ARC-Easy [53]、ARC-Challenge [53]、Hellaswag [54]、Winogrande [55]、PIQA [56]、およびOpenbookQA [57]を含む一連の言語タスクで評価しました。結果は表1に示されています。データセットの詳細は付録Cにあります。すべての評価はLM評価ハーネス[58]を使用して実行されます。MatMulフリーLMモデルは、すべてのタスクでTransformer++ベースラインと比較して競争力のあるパフォーマンスを達成し、MatMul操作がないにもかかわらずゼロショット学習における効果を示し、三元重みから必要なメモリが少ないことを示しています。特に、2.7B MatMulフリーLMモデルはARC-ChallengeとOpenbookQAでTransformer++の対応モデルを上回り、他のタスクでは同等のパフォーマンスを維持しています。モデルサイズが大きくなるにつれて、MatMulフリーLMとTransformer++の間のパフォーマンスギャップは狭まり、これはスケーリング法則と一致しています。これらの結果は、MatMulフリーアーキテクチャが質問応答や常識推論から物理理解まで、多様な言語タスクで強力なゼロショットパフォーマンスを達成できることを強調しています。
-

Fig. 4(d) presents a comparison of GPU inference memory consumption and latency between the
proposed MatMul-free LM and Transformer++ for various model sizes. In the MatMul-free LM, we
employ BitBLAS [60] for acceleration to further improve efficiency. The evaluation is conducted
with a batch size of 1 and a sequence length of 2048. The MatMul-free LM consistently demonstrates
lower memory usage and latency compared to Transformer++ across all model sizes. For a single
layer, the MatMul-free LM requires only 0.12 GB of GPU memory and achieves a latency of 3.79 ms,
while Transformer++ consumes 0.21 GB of memory and has a latency of 13.87 ms. As the model
size increases, the memory and latency advantages of the MatMul-free LM become more pronounced.
It is worth noting that for model sizes larger than 2.7B, the results are simulated using randomly
initialized weights. For the largest model size of 13B parameters, the MatMul-free LM uses only
4.19 GB of GPU memory and has a latency of 695.48 ms, whereas Transformer++ requires 48.50
GB of memory and exhibits a latency of 3183.10 ms. These results highlight the efficiency gains
achieved by the MatMul-free LM, making it a promising approach for large-scale language modeling
tasks, particularly during inference.
---
図4(d)は、提案されたMatMulフリーLMとTransformer++のGPU推論メモリ消費とレイテンシの比較を示しています。MatMulフリーLMでは、BitBLAS [60]を使用して効率をさらに向上させます。評価はバッチサイズ1、シーケンス長2048で実施されます。MatMulフリーLMは、すべてのモデルサイズでTransformer++と比較して、一貫して低いメモリ使用量とレイテンシを示します。単一層の場合、MatMulフリーLMはGPUメモリを0.12 GBのみ必要とし、レイテンシは3.79 msですが、Transformer++は0.21 GBのメモリを消費し、レイテンシは13.87 msです。モデルサイズが大きくなるにつれて、MatMulフリーLMのメモリとレイテンシの利点はより顕著になります。2.7Bを超えるモデルサイズでは、結果はランダムに初期化された重みを使用してシミュレーションされていることに注意してください。最大モデルサイズの13Bパラメータでは、MatMulフリーLMはGPUメモリを4.19 GBのみ使用し、レイテンシは695.48 msですが、Transformer++は48.50 GBのメモリを必要とし、レイテンシは3183.10 msです。これらの結果は、MatMulフリーLMによって達成された効率向上を強調し、大規模な言語モデリングタスク、特に推論時に有望なアプローチとなることを示しています。
-

We evaluate our proposed Fused BitLinear and Vanilla BitLinear implementations in terms of training
time and memory usage, shown in Fig.4(a-b). For each experiment, we set the input size and sequence
length to 1024. All experiments are conducted using an NVIDIA A100 80GB GPU. Note that during
training, the sequence length and batch dimensions are flattened, making the effective batch size the
product of these dimensions.
Our experiments show that our fused operator benefits from larger batch sizes in terms of faster
training speeds and reduced memory consumption. When the batch size is 2
8
, the training speed
of the 1.3B parameter model improves from 1.52s to 1.21s per iteration, a 25.6% speedup over the
vanilla implementation. Additionally, memory consumption decreases from 82GB to 32GB, a 61.0%
reduction in memory usage. The performance of the Fused implementation improves significantly
with larger batch sizes, allowing more samples to be processed simultaneously and reducing the total
number of iterations.
---
提案されたFused BitLinearとVanilla BitLinearの実装を、トレーニング時間とメモリ使用量の観点から評価します。図4(a-b)に示されています。各実験では、入力サイズとシーケンス長を1024に設定しています。すべての実験はNVIDIA A100 80GB GPUを使用して実施されます。トレーニング中は、シーケンス長とバッチ次元がフラット化されるため、効果的なバッチサイズはこれらの次元の積になります。
私たちの実験では、フューズドオペレーターが大きなバッチサイズでトレーニング速度の向上とメモリ消費の削減に利益をもたらすことを示しています。バッチサイズが2^8のとき、1.3Bパラメータモデルのトレーニング速度は1.52秒から1.21秒に改善され、バニラ実装に比べて25.6%の速度向上を達成します。さらに、メモリ消費は82GBから32GBに減少し、メモリ使用量が61.0%削減されます。フューズド実装のパフォーマンスは、バッチサイズが大きくなるにつれて大幅に改善され、より多くのサンプルを同時に処理できるようになり、総イテレーション数が減少します。
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
---
回路の複雑さを削減する方法の1つは、計算を簡素化しながら精度を維持できる量子化された値を使用することです[5]。したがって、リザーバ層の出力[式(1)]を量子化された重みを使用して計算しました。一般に、入力およびリザーバ層の重みは実数であり、実数乗算を計算するためにいくつかのDSPが必要です。したがって、実数値の重みを三元値（0または±1）に変換しました。さらに、この量子化をトレーニングモードと予測モードの両方で使用することで精度が維持されます。
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
再帰型ニューラルネットワーク（RNN）は、シーケンスデータの処理に優れたパフォーマンスを示しています。ただし、再帰的な性質のため、複雑でメモリ集約的です。これらの制限により、RNNはハードウェアリソースが限られたリアルタイムプロセスを必要とするモバイルデバイスに埋め込むことが困難になります。上記の問題に対処するために、トレーニングフェーズ中にバイナリおよび三元重みを学習できる方法を導入し、RNNのハードウェア実装を容易にします。その結果、このアプローチを使用すると、すべての乗算累積操作が単純な累積に置き換えられ、シリコン面積と電力消費の点でカスタムハードウェアに大きな利点をもたらします。ソフトウェア側では、長短期記憶（LSTM）およびゲート付き再帰ユニット（GRU）を使用して、シーケンス分類や言語モデリングなどのさまざまなシーケンシャルモデルでの精度（パフォーマンス）を評価します。私たちの方法は、ランタイム中にバイナリ/三元重みを使用しながら、前述のタスクで競争力のある結果を達成できることを示しています。ハードウェア側では、バイナリ/三元重みを持つLSTMの再帰計算を加速するカスタムハードウェアを提示します。最終的に、バイナリ/三元重みを持つLSTMは、フル精度ハードウェア実装設計と比較して最大12倍のメモリ節約と10倍の推論速度向上を達成できることを示します。
-

• ARC-Easy and ARC-Challenge [53]: Question answering datasets that require models to
demonstrate reasoning and knowledge acquisition abilities. ARC-Easy contains questions
that are straightforward to answer, while ARC-Challenge includes more difficult questions.
• Hellaswag [54]: A commonsense inference dataset that tests a model’s ability to choose
the most plausible continuation of a given context. The dataset is constructed from a large
corpus of movie scripts and requires models to have a deep understanding of everyday
situations.
• Winogrande [55]: A benchmark for measuring a model’s ability to perform commonsense
reasoning and coreference resolution. The dataset consists of carefully constructed minimal
pairs that require models to use commonsense knowledge to resolve ambiguities.
• PIQA [56]: A benchmark for physical commonsense reasoning that tests a model’s understanding of physical properties, processes, and interactions. The dataset contains multiplechoice questions that require models to reason about physical scenarios.
• OpenbookQA [57]: A question answering dataset that measures a model’s ability to combine
scientific facts with commonsense reasoning. The dataset is constructed from a set of science
questions and a collection of scientific facts, requiring models to use the provided facts to
answer the questions.
---
• ARC-EasyおよびARC-Challenge [53]：モデルが推論能力と知識獲得能力を示す必要がある質問応答データセット。ARC-Easyには簡単に回答できる質問が含まれ、ARC-Challengeにはより難しい質問が含まれています。
• Hellaswag [54]：与えられたコンテキストの最も妥当な継続を選択するモデルの能力をテストする常識推論データセット。このデータセットは映画の脚本の大規模なコーパスから構築され、日常的な状況に対する深い理解をモデルに要求します。
• Winogrande [55]：モデルの常識推論能力と照応解決能力を測定するベンチマーク。データセットは、モデルが常識知識を使用してあいまいさを解決する必要がある慎重に構築された最小ペアで構成されています。
• PIQA [56]：物理的な常識推論のベンチマークで、モデルの物理的特性、プロセス、および相互作用の理解をテストします。このデータセットには、モデルが物理シナリオについて推論する必要がある選択式の質問が含まれています。
• OpenbookQA [57]：科学的事実と常識推論を組み合わせるモデルの能力を測定する質問応答データセット。このデータセットは、科学的質問のセットと科学的事実のコレクションから構築され、モデルが提供された事実を使用して質問に回答する必要があります。s
-

