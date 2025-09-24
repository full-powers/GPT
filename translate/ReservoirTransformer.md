# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。

We demonstrate that transformers obtain impressive performance even when some of the
layers are randomly initialized and never updated. Inspired by old and well-established
ideas in machine learning, we explore a variety
of non-linear “reservoir” layers interspersed
with regular transformer layers, and show improvements in wall-clock compute time until
convergence, as well as overall performance,
on various machine translation and (masked)
language modelling tasks.
---
我々は、トランスフォーマーが一部の層がランダムに初期化され、更新されない場合でも印象的なパフォーマンスを発揮することを示します。機械学習における古くからの確立されたアイデアに触発され、様々な非線形の「リザーバー」層を通常のトランスフォーマー層と交互に配置し、収束までのウォールクロック計算時間と全体的なパフォーマンスの改善を、様々な機械翻訳および（マスクされた）言語モデリングタスクで示します。
---

We introduce a area under the convergence
curve metric for measuring performanceefficiency trade-offs, and show that replacing
regular transformer layers with reservoir layers leads to improvements.
• We show that the addition of reservoir layers
leads to improved test set generalization on a
variety of tasks in a variety of settings.
• We show that pre-trained masked language modelling architectures like BERT and
RoBERTa (Liu et al., 2019) can benefit from
having some of their layers frozen, both during pre-training as well as when fine-tuning
on downstream tasks.
• We experiment with different types of reservoir layers, including convolutional and recurrent neural network-based ones.
• We show empirical evidence that the backward pass can be skipped in its entirety by
approximating upstream gradients using an
approach we call backskipping, which can
reduce the training compute further without
sacrificing performance.
---
我々は、収束曲線の下の面積メトリックを導入し、パフォーマンスと効率のトレードオフを測定します。通常のトランスフォーマー層をリザーバー層に置き換えることで改善が得られることを示します。
• リザーバー層の追加により、様々なタスクにおいて、様々な設定でテストセットの一般化が改善されることを示します。
• BERTやRoBERTa（Liu et al., 2019）のような事前学習されたマスク付き言語モデリングアーキテクチャは、事前学習中およびダウンストリームタスクのファインチューニング時に、いくつかの層を凍結することで利益を得ることができることを示します。
• 畳み込みや再帰型ニューラルネットワークベースのリザーバー層など、さまざまなタイプのリザーバー層を実験します。
•上流の勾配を近似することで、バックパス全体をスキップできるという経験的証拠を示します。このアプローチは、バックスキッピングと呼ばれ、パフォーマンスを犠牲にすることなく、
トレーニング計算をさらに削減できます。
---

Now, during every “backward pass”, we compute the Jacobian for parameters θ
L at layer L,
which are used to update the parameters of L, θ
L
t
,
as well as to compute the next layer’s Jacobian,
thus back-propagating the gradients. In this work
however, for some of the layers, we still backpropagate through them to compute gradients for earlier layers, but we never apply the parameter update. As a result, these layers stay fixed at their
initialization, saving computational resources.
---
現在、すべての「バックワードパス」で、層Lのパラメータθ
Lのヤコビアンを計算し、これらは層Lのパラメータθ
L
tを更新するために使用され、次の層のヤコビアンを計算するためにも使用されます。これにより、勾配が逆伝播されます。しかし、この研究では、一部の層に対しては、以前の層の勾配を計算するために逆伝播を行いますが、パラメータの更新は決して適用しません。その結果、これらの層は初期化時のまま固定され、計算リソースを節約します。
---

This work explores inserting random non-linear
transformations, or what we call reservoir layers,
into transformer networks. Specifically, we experiment with a variety of reservoir layers:
• Transformer Reservoir: The standard transformer layer as described above, but with all
parameters fixed after initialization, including the self-attention module.
• FFN Reservoir: A transformer-style fixed
feed-forward layer without any self-attention,
i.e., FFN(LayerNorm(Previous layer)) +
Previous layer.
• BiGRU Reservoir: A fixed bidirectional
Gated Recurrent Unit (Cho et al., 2014) layer,
which is closer in spirit to previous work on
reservoir computing, most of which builds on
recurrent neural network architectures.
• CNN Reservoir: A fixed Convolutional Neural Network (LeCun et al., 1998) layer,
specifically light dynamical convolution layers (Wu et al., 2019), which are known to be
competitive with transformers in sequenceto-sequence tasks.
We find that all these approaches work well, to
a certain extent. For clarity, we focus primarily on
the first two reservoir layers, but include a broader
comparison in Appendix A.
In each case, contrary to traditional reservoir
computing, our reservoir layers are interspersed
throughout a regular transformer network, or what
we call a reservoir transformer. Since random projections are not learned and might introduce noise,
subsequent normal transformer “readout” layers
might be able to benefit from additional depth
while allowing us to recover from any adverse effects of randomness. For example, previous work
has shown that ResNets, with all of their parameters fixed except for the scale and shift parameters
of batch normalization, can still achieve high performance, simply by scaling and shifting random
features (Frankle et al., 2020). Adding some form
of noise to the parameters is also known to help
convergence and generalization (Jim et al., 1995,
1996; Gulcehre et al., 2016; Noh et al., 2017).
---
この研究では、ランダムな非線形変換、またはリザーバー層と呼ばれるものをトランスフォーマーネットワークに挿入することを探求します。具体的には、さまざまなリザーバー層を実験します：
• トランスフォーマーリザーバー：上記で説明した標準のトランスフォーマー層ですが、自己注意モジュールを含むすべてのパラメータが初期化後に固定されています。
• FFNリザーバー：自己注意のないトランスフォーマースタイルの固定フィードフォワード層、つまりFFN（LayerNorm（前の層））+前の層。
• BiGRUリザーバー：固定双方向ゲート付きリカレントユニット（Cho et al., 2014）層で、これはリザーバーコンピューティングに関する以前の研究に近いものであり、その多くはリカレントニューラルネットワークアーキテクチャに基づいています。
• CNNリザーバー：固定畳み込みニューラルネットワーク（LeCun et al., 1998）層、
具体的には、シーケンスツーシーケンスタスクでトランスフォーマーと競合することが知られているライトダイナミカル畳み込み層（Wu et al., 2019）。
これらのアプローチはすべて、ある程度うまく機能することがわかります。明確にするために、主に最初の2つのリザーバー層に焦点を当てますが、付録Aでより広範な比較を含めます。
各ケースにおいて、従来のリザーバーコンピューティングとは対照的に、リザーバー層は通常のトランスフォーマーネットワーク全体に散在しており、リザーバートランスフォーマーと呼ばれるものです。ランダムな投影は学習されず、ノイズを導入する可能性があるため、後続の通常のトランスフォーマー「リードアウト」層は、ランダム性の悪影響から回復しながら、追加の深さの恩恵を受けることができるかもしれません。たとえば、以前の研究では、バッチ正規化のスケールとシフトパラメータムを除いてすべてのパラメータが固定されたResNetが、ランダムな特徴を単にスケーリングおよびシフトすることで高いパフォーマンスを達成できることが示されています（Frankle et al., 2020）。パラメータに何らかの形のノイズを追加することも、収束と一般化に役立つことが知られています（Jim et al., 1995, 1996; Gulcehre et al., 2016; Noh et al., 2017）。
---

We evaluate on IWSLT de-en (Cettolo et al., 2015)
and WMT en-de (Bojar et al., 2014) for machine translation; enwiki8 (LLC, 2009) for language modelling; and experiment with RoBERTa
(Liu et al., 2019) in our pretraining experiments.
For IWSLT, we follow the pre-processing steps
in Edunov et al. (2018). The train/val/test split
is 129k/10k/6.8k sentences. For WMT, we follow pre-process as in Ott et al. (2018), with
4.5M/16.5k/3k sentences in train/val/test. For enwiki8, we follow the pre-processing steps in Dai
et al. (2019). The train/val/test split is 1M/54k/56k
sentences. For RoBERTa pretraining, we follow
the pre-processing steps in Liu et al. (2019).
We use 8 Volta V100 GPUs for WMT and enwik8, 32 V100 GPUs for RoBERTa and a single V100 for IWSLT. The hyperparameters for
IWSLT14 and WMT16 were set to the bestperforming values from Ott et al. (2018) and Kasai
et al. (2020) respectively. The enwik8 experiment
settings followed Bachlechner et al. (2020) and the
RoBERTa experiments followed Liu et al. (2019).
All the experiments in this paper were run with
3 random seeds and the mean and standard deviation are reported. For the relatively small IWSLT,
the Tˆ value in the AUCC metric was set to 4 hours.
For the larger WMT, we set it to 20 hours. For
enwiki8, it was 30 hours; and for the RoBERTa
pre-training experiments, it was set to 60 hours.
The projection weights in random layers were
initialized using orthogonal initialization (Saxe
et al., 2013), since random orthogonal projections should ideally be maximally informationpreserving, and which was found to work well empirically for initializing fixed random representations in previous work (Wieting and Kiela, 2019).
Biases and layer norm parameters were initialized
using their respective PyTorch defaults (based on
Xavier init; Glorot and Bengio, 2010).
We intersperse reservoir layers in alternating
fashion starting from the middle. Specifically, we
alternate one reservoir layer with one transformer
layer, and place the alternating block in the middle. For example: a 7-layer encoder LLLLLLL
in which we replace three layers with reservoirs becomes LRLRLRL, and with two becomes
LLRLRLL. See Appendix C for a study comparing this strategy to alternative approaches (e.g.,
freezing in the bottom, middle or top).
---
我々は、IWSLT de-en (Cettolo et al., 2015)とWMT en-de (Bojar et al., 2014)を機械翻訳のために評価し、言語モデリングのためにenwiki8 (LLC, 2009)を使用し、事前学習実験ではRoBERTa (Liu et al., 2019)を実験します。IWSLTについては、Edunov et al. (2018)の前処理手順に従います。トレイン/バリデーション/テストの分割はそれぞれ129k/10k/6.8k文です。WMTについては、Ott et al. (2018)の前処理に従い、トレイン/バリデーション/テストでそれぞれ4.5M/16.5k/3k文です。enwiki8については、Dai et al. (2019)の前処理手順に従います。トレイン/バリデーション/テストの分割は1M/54k/56k文です。RoBERTaの事前学習については、Liu et al. (2019)の前処理手順に従います。
WMTとenwiki8には8台のVolta V100 GPUを使用し、
RoBERTaには32台のV100 GPU、IWSLTには1台のV100を使用します。IWSLT14とWMT16のハイパーパラメータは、それぞれOtt et al. (2018)とKasai et al. (2020)の最適な値に設定しました。enwiki8の実験設定はBachlechner et al. (2020)に従い、RoBERTaの実験はLiu et al. (2019)に従いました。
この論文のすべての実験は3つのランダムシードで実行され、平均と標準偏差が報告されています。比較的小さなIWSLTでは、AUCCメトリックのTˆ値は4時間に設定されました。より大きなWMTでは、20時間に設定しました。enwiki8では30時間、RoBERTaの事前学習実験では60時間に設定されました。
ランダム層の投影重みは、直交初期化（Saxe et al., 2013）を使用して初期化されました。これは、ランダムな直交投影が理想的には最大限に情報を保持するべきであり、以前の研究（Wieting and Kiela, 2019）で固定ランダム表現の初期化にうまく機能することが実証されているためです。
バイアスとレイヤーノルムパラメータは、それぞれのPyTorchデフォルト（Xavier初期化に基づく; Glorot and Bengio, 2010）を使用して初期化されました。
リザーバー層は、中央から交互に配置しています。具体的には、1つのリザーバー層と1つのトランスフォーマー層を交互に配置し、交互のブロックを中央に配置します。例えば、3つの層をリザーバーに置き換えた7層エンコーダーLLLLLLLはLRLRLRLになり、2つの場合はLLRLRLLになります。代替アプローチ（例えば、下部、中間、または上部での凍結）と比較する研究については、付録Cを参照してください。
---

In what follows, we first show our main result, on
a variety of tasks: reservoir transformers mostly
have better AUCC metrics; less training time per
epoch; less convergence time until the best validation performance is achieved; and even improved
test set generalization metrics. As a strong baseline method, we compare to LayerDrop (Fan et al.,
2019). LayerDrop can also be seen as a method
that dynamically bypasses parts of the computation during Transformer training in an attempt to
improve efficiency, and making it a strong comparison to examine our methods. Then, we examine
whether we can minimize the expectation over the
gradients of upstream layers in the network such
that we do not at all have to pass gradients through
the reservoir layers, skipping their backward pass.
---
以下では、さまざまなタスクにおける主な結果を示します。リザーバートランスフォーマーは、主にAUCCメトリックが優れており、エポックごとのトレーニング時間が短く、最良の検証パフォーマンスが達成されるまでの収束時間も短く、テストセットの一般化メトリックも改善されています。強力なベースライン手法として、LayerDrop（Fan et al., 2019）と比較します。LayerDropは、トランスフォーマーのトレーニング中に計算の一部を動的にバイパスする方法とも見なすことができ、効率を改善しようとする試みであり、我々の手法を検証するための強力な比較となります。その後、ネットワーク内の上流層の勾配に対する期待値を最小化できるかどうかを調べます。これにより、リザーバー層を通じて勾配を全く通過させず、そのバックワードパスをスキップすることができます。
---

Table 1 and 2 show the time it took to achieve
the maximum validation BLEU score and how that
relates to the regular transformer, demonstrating
that reservoir transformers consistently converge
faster in terms of wall-clock time. We save up
to 22% convergence wall-clock time using reser-
voir transformers as much with the same number
of updateable layers. We save as much as 27%
time until convergence a 24 layer model on WMT,
as shown in Table 2. One other noticeable point
is that we can see that the T Reservoir achieves
similar performance to LayerDrop on IWSLT and
WMT in terms of wall-clock per epoch and wall-
clock time to the best performance. However, on
both tasks, FFN Reservoir performs much better
than LayerDrop in terms of efficiency per epoch
and achieves better/similar performance in less
time in each case. As a point of reference, a half
hour gain on IWSLT would translate to a gain of
several days in the training of bigger transformer
models like GPT-3 (Brown et al., 2020).
---
表1と2は、最大の検証BLEUスコアを達成するのにかかった時間と、それが通常のトランスフォーマーとどのように関連しているかを示しており、リザーバートランスフォーマーがウォールクロック時間の観点から一貫してより速く収束することを示しています。更新可能な層の数が同じである場合、リザーバートランスフォーマーを使用することで、最大22％の収束ウォールクロック時間を節約できます。表2に示すように、WMTで24層モデルが収束するまでに最大27％の時間を節約できます。もう一つ注目すべき点は、TリザーバーがIWSLTとWMTでLayerDropと同様のパフォーマンスを達成していることです。これはエポックごとのウォールクロック時間と最良のパフォーマンスまでのウォールクロック時間に関してです。しかし、両方のタスクで、FFNリザーバーはエポックごとの効率においてLayerDropよりもはるかに優れており、各ケースでより少ない時間でより良い/同等のパフォーマンスを達成しています。参考までに、IWSLTで30分の利得は、GPT-3（Brown et al., 2020）などの大規模なトランスフォーマーモデルのトレーニングで数日の利得に相当します。
---

Figure 2 shows the results of fine-tuning. We
observe that the reservoir transformer outperforms
normal RoBERTa at all depths in both tasks. At
lower depth, the improvements are substantial. As
a sanity check, we also experiment with freez-
ing some of the layers in a regular pre-trained
RoBERTa model during fine-tuning only (Trans-
former “frozen finetuned” in the Figure) and show
that this helps a little but is still outperformed by
the reservoir transformer.
---
図2はファインチューニングの結果を示しています。リザーバートランスフォーマーは、両方のタスクで通常のRoBERTaをすべての深さで上回ることを観察します。低い深さでは、改善はかなりのものです。サニティチェックとして、ファインチューニング中に通常の事前学習済みRoBERTaモデルの一部の層を凍結する実験も行い（図の「凍結ファインチューニングされたトランスフォーマー」）、これが少しは役立つことを示しますが、リザーバートランスフォーマーにはまだ劣ることがわかります。
---

Machine translation (MT) is one of the core
tasks of NLP. We demonstrate on two well-known
MT datasets, IWSLT’14 German-English and
WMT’16 English-German, that reservoir trans-
formers obtain a better AUCC. For the raw vali-
dation plots over time that were used to calculate
the AUCC, please refer to Appendix F.
Following Kasai et al. (2020), the architecture
of the network is an N-layer reservoir transformer
encoder, followed by a regular shallow one- or
two-layer decoder. This design choice has been
shown to lead to very good speed and efficiency
trade-offs, and serves as a good baseline for our
experiments. Moreover, shallow decoders make it
easier to decide where to place reservoir layers (in
the encoder) and makes it more straightforward to
identify where performance gains come from.
Figure 1 shows the results for IWSLT (left) and
WMT (middle). On the y-axis we show valida-
tion AUCC for the BLEU metric; on the x-axis
we show the number of updatable layers in the en-
coder. The performance of a regular transformer
encoder with 6 layers and a reservoir transformer
encoder with 6 layers plus N additional reservoir
layers are plotted for the same x-axis value to
show the total number of updated layers. Plots
for the total number of layers (updatable plus not-
updatable, so essentially shifted versions of the
plots) are shown in Appendix E.
WMT is much larger and requires a much
deeper encoder, as illustrated by the fact that a
certain minimum depth is required for reservoir
transformers to achieve a comparable validation
AUCC. At test time, reservoir transformers outper-
form regular transformers for almost all encoder
depths. The FFN Reservoir seems to work best
in both cases, which is surprising because it does
not have any self-attention component at all. This
finding shows that self-attention, or the mecha-
nism to summarize context information, should be
learned if present. Once the context features have
been gathered, a random projection via a fixed
FFN module appears to be beneficial.
---
機械翻訳（MT）はNLPの中核的なタスクの一つです。IWSLT’14ドイツ語-英語とWMT’16英語-ドイツ語の2つのよく知られたMTデータセットで、リザーバートランスフォーマーがより良いAUCCを得ることを示します。AUCCを計算するために使用された時間経過に伴う生の検証プロットについては、付録Fを参照してください。
Kasai et al. (2020)に従い、ネットワークのアーキテクチャはN層のリザーバートランスフォーマーエンコーダーで、その後に通常の浅い1層または2層のデコーダ  ーが続きます。この設計選択は、非常に良い速度と効率のトレードオフをもたらすことが示されており、我々の実験の良いベースラインとなります。さらに、浅いデコーダーはリザーバー層を配置する場所を決定しやすくし、パフォーマンス向上の原因を特定しやすくします。
図1はIWSLT（左）とWMT（中央）の結果を示しています。y軸にはBLEUメトリックの検証AUCCを示し、
x軸にはエンコーダーの更新可能な層の数を示しています。6層の通常のトランスフォーマーエンコーダーと
6層のリザーバートランスフォーマーエンコーダーに
N個の追加のリザーバー層を持つエンコーダーのパフォーマンスが同じx軸値でプロットされ、更新された層の総数を示しています。総層数（更新可能な層と更新不可能な層、つまり基本的にプロットのシフトバージョン）のプロットは付録Eに示されています。
WMTははるかに大きく、より深いエンコーダーが必要です。これは、リザーバのサイズは32です。
層トランスフォーマーが同等の検証AUCCを達成するために特定の最小深度が必要であることからも明らかです。テスト時には、リザーバートランスフォーマーはほとんどすべてのエンコーダー深度で通常のトランスフォーマーを上回ります。FFNリザーバーは両方のケースで最も効果的に機能するようで、これは自己注意コンポーネントが全くないため驚きです。この発見は、自己注意、つまりコンテキスト情報を要約するメカニズムは、存在する場合には学習されるべきであることを示しています。一度コンテキスト特徴が収集されると、固定FFNモジュールを介したランダムな投影が有益であるようです。
---