# 英文を日本語に翻訳してください。
# 英文の最後に---とつけるのでそこまでを翻訳し、日本語の文章を続けてください。

As an efficient recurrent neural network (RNN)
model, reservoir computing (RC) models, such as Echo State
Networks, have attracted widespread attention in the last decade.
However, while they have had great success with time series data
[1], [2], many time series have a multiscale structure, which a
single-hidden-layer RC model may have difficulty capturing. In
this paper, we propose a novel hierarchical reservoir computing
framework we call Deep Echo State Networks (Deep-ESNs). The
most distinctive feature of a Deep-ESN is its ability to deal
with time series through hierarchical projections. Specifically,
when an input time series is projected into the high-dimensional
echo-state space of a reservoir, a subsequent encoding layer
(e.g., a PCA, autoencoder, or a random projection) can project
the echo-state representations into a lower-dimensional space.
These low-dimensional representations can then be processed by
another ESN. By using projection layers and encoding layers
alternately in the hierarchical framework, a Deep-ESN can not
only attenuate the effects of the collinearity problem in ESNs,
but also fully take advantage of the temporal kernel property of
ESNs to explore multiscale dynamics of time series. To fuse the
multiscale representations obtained by each reservoir, we add
connections from each encoding layer to the last output layer.
Theoretical analyses prove that stability of a Deep-ESN is guaranteed by the echo state property (ESP), and the time complexity
is equivalent to a conventional ESN. Experimental results on some
artificial and real world time series demonstrate that Deep-ESNs
can capture multiscale dynamics, and outperform both standard
ESNs and previous hierarchical ESN-based models.
---
効率的なリカレントニューラルネットワーク（RNN）モデルとして、エコー状態ネットワーク（ESN）などのリザーバーコンピューティング（RC）モデルは、過去10年間で広く注目を集めています。しかし、時系列データにおいては大きな成功を収めている一方で[1]、[2]、多くの時系列データはマルチスケール構造を持ち、単一の隠れ層RCモデルでは捉えきれないことがあります。本論文では、Deep Echo State Networks（Deep-ESN）と呼ばれる新しい階層型リザーバーコンピューティングフレームワークを提案します。Deep-ESNの最も特徴的な点は、階層的な投影を通じて時系列データを扱う能力です。具体的には、入力時系列がリザーバーの高次元エコー状態空間に投影されると、その後のエンコード層（例えば、PCA、自動エンコーダー、またはランダム投影）がエコー状態表現を低次元空間に投影できます。これらの低次元表現は別のESNによって処理されます。階層型フレームワークで投影層とエンコード層を交互に使用することで、Deep-ESNはESNにおける共線性問題の影響を軽減するだけでなく、ESNの時間カーネル特性を最大限に活用して時系列データのマルチスケールダイナミクスを探求できます。各リザーバーから得られたマルチスケール表現を融合するために、各エンコード層から最後の出力層への接続を追加します。理論的な分析により、Deep-ESNの安定性はエコー状態特性（ESP）によって保証され、時間計算量は従来のESNと同等であることが証明されています。いくつかの人工および実世界の時系列データに関する実験結果は、Deep-ESNがマルチスケールダイナミクスを捉えることができ、標準のESNや以前の階層型ESNベースのモデルを上回ることを示しています。
---

The main contributions
of this paper can be summarized as follows.
1) We propose a novel multiple projection-encoding deep
reservoir computing framework called Deep-ESN, which
bridges the gap between reservoir computing and deep
learning.
2) By unsupervised encoding of echo states and adding
direct information flow from each encoder layer to
the last output layer, the proposed Deep-ESN can not
only obtain multiscale dynamics, but also dramatically
attenuate the effects of collinearity problem in ESNs.
3) In a theoretical analysis, we analyze the stability and
computational complexity of Deep-ESN. We also verify
that the collinearity problem is alleviated with small
condition numbers, and that each layer of the Deep-ESN
can capture various dynamics of time series.
4) Compared with the several RC hierarchies, Deep-ESN
achieves better performances on well-known chaotic
dynamical system tasks and some real world time series.
---
本論文の主な貢献は以下の通りです。
1) リザーバーコンピューティングと深層学習のギャップを埋める、Deep-ESNと呼ばれる新しい複数投影エンコード深層リザーバーコンピューティングフレームワークを提案します。
2) エコー状態の教師なしエンコードと、各エンコーダ層から最後の出力層への直接情報フローを追加することで、提案されたDeep-ESNはマルチスケールダイナミクスを取得できるだけでなく、ESNにおける共線性問題の影響を劇的に軽減することができます。
3) 理論的な分析において、Deep-ESNの安定性と計算複雑性を分析します。また、共線性問題が小さな条件数で緩和されることを検証し、Deep-ESNの各層が時系列のさまざまなダイナミクスを捉えることができることを示します。
4) いくつかのRC階層と比較して、Deep-ESNは有名なカオスダイナミカルシステムタスクやいくつかの実世界の時系列においてより良いパフォーマンスを達成します。
---

The important hyperparameters used for initializing an ESN
are IS - the input scaling, SR - the spectral radius, α - the
sparsity and the aforementioned leak rate γ.
1) IS is used for the initialization of the matrix Win: the
elements of Win obey the uniform distribution of -IS
to IS.
2) SR is the spectral radius of Wres, given by
Wres = SR ·
W
λmax(W)
(4)
where λmax(W) is the largest eigenvalue of matrix W
and the elements of W are generated randomly from
[−0.5, 0.5]. To satisfy the Echo State Property (ESP)
[10], [29], SR should be set smaller than 1. This is a
necessary condition of ESN stability. The ESP will be
discussed in more detail later.
3) α denotes the proportion of non-zero elements in Wres
.
We set α to 0.1.
In short, ESNs have a very simple training procedure,
and due to the high-dimensional projection and highly sparse
connectivity of neurons in the reservoir, it has abundant nonlinear echo states and short-term memory, which are very
useful for modeling dynamical systems. However, a single
ESN can not deal with input signals that require complex hierarchical processing and it cannot explicitly support multiple
time scales. In the next section, we will propose a novel deep
reservoir computing framework to resolve this issue.
---
ESNの初期化に使用される重要なハイパーパラメータは、IS - 入力スケーリング、SR - スペクトル半径、α - スパース性、前述の漏れ率γです。
1) ISは、行列Winの初期化に使用されます。Winの要素は、-ISからISの一様分布に従います。
2) SRはWresのスペクトル半径であり、次のように与えられます。
Wres = SR ·
W
λmax(W)
(4)
ここで、λmax(W)は行列Wの最大固有値であり、Wの要素は[−0.5, 0.5]からランダムに生成されます。エコー状態特性（ESP）[10]、[29]を満たすために、SRは1未満に設定する必要があります。これはESNの安定性のための必要条件です。ESPについては後で詳しく説明します。
3) αはWresの非ゼロ要素の割合を示します。
αは0.1に設定します。
要するに、ESNは非常にシンプルなトレーニング手順を持ち、高次元の投影とリザーバ内のニューロンの高いスパース接続性により、豊富な非線形エコー状態と短期記憶を持ち、動的システムのモデル化に非常に役立ちます。しかし、単一のESNは、複雑な階層処理を必要とする入力信号に対処できず、明示的に複数の時間スケールをサポートすることもできません。次のセクションでは、この問題を解決するための新しい深層リザーバーコンピューティングフレームワークを提案します。
---

In this section, the details of the proposed Deep-ESN will
be described, as well as the related analysis of stability and
computational complexity.
Although the main idea of hierarchical ESN-based models is
to capture multiscale dynamics of time series by constructing
a deep architecture, there are three main features of our DeepESN that contrast with previous approaches:
• Multiple Projection-Encoding: Instead of directly stacking multiple ESNs, Deep-ESN uses the encoder layer between reservoirs. In this way, the DeepESN can not only
obtain abundant multiscale dynamical representations of
inputs by fully taking advantage of high-dimensional
projections, but also solves the collinearity problem in
ESNs.
• Multiscale feature fusion: In order to better fuse multiscale dynamical representations captured by each reservoir, we add connections (called feature links) from each
encoding layer to the last output layer.
• Simplicity of Training: Unlike some previous hierarchical
ESN-based models, training the whole model layer by
layer, the only trainable layer of the Deep-ESN is the last
output layer, which retains the efficient computation of
RC without relying on gradient-propagation algorithms.
---
このセクションでは、提案されたDeep-ESNの詳細と、安定性と計算複雑性に関する関連分析について説明します。
階層型ESNベースのモデルの主なアイデアは、深層アーキテクチャを構築することで時系列のマルチスールダイナミクスを捉えることですが、Deep-ESNには以前のアプローチと対照的な3つの主な特徴があります。
•複数の投影エンコード：複数のESNを直接スタックするのではなく、Deep-ESNはリザーバー間にエンコーダ層を使用します。この方法により、Deep-ESNは高次元の投影を最大限に活用して入力の豊富なマルチスールダイナミクス表現を取得できるだけでなく、ESNの共線性問題も解決します。
•マルチスール特徴融合：各リザーバーが捉えたマルチスールダイナミクス表現をより適切に融合するために、各エンコーディング層から最後の出力層への接続（特徴リンクと呼ばれる）を追加します。
•トレーニングの簡素化：以前のいくつかの階層型ESNベースのモデルとは異なり、Deep-ESNの唯一のトレーニング可能な層は最後の出力層であり、勾配伝播アルゴリズムに依存せずにRCの効率的な計算を保持します。
---

Although Deep-ESN is a deep neural model of reservoir
computing, there are not large additional costs in the whole
learning process. In this section, we analyze the computational
complexity of Deep-ESN.
---
Deep-ESNはリザーバーコンピューティングの深層ニューラルモデルですが、全体の学習プロセスにおいて大きな追加コストはありません。このセクションでは、Deep-ESNの計算複雑性を分析します。
---

Hierarchical multiscale structures naturally exist in many
temporal data, a phenomenon that is difficult to capture
by a conventional ESN. To overcome this limitation, we
propose a novel hierarchical reservoir computing framework
called Deep-ESNs. Instead of directly stacking reservoirs, we
combine the randomly-generated reservoirs with unsupervised
encoders, retaining the high-dimensional projection capacity as
well as the efficient learning of reservoir computing. Through
this multiple projection-encoding system, we not only alleviate the collinearity problem in ESNs, but we also capture
the multiscale dynamics in each layer. The feature links in
our Deep-ESN provides multiscale information fusion, which
improves the ability of the network to fit the time series.
We also presented a derivation of the stability condition and
the computational complexity of our Deep-ESN. The results
show that our Deep-ESN with efficient unsupervised encoders
(e.g., PCA) can be as efficiently learned as a shallow ESN,
retaining the major computational advantages of traditional
reservoir-computing networks.
---
多くの時間データには階層的なマルチスケール構造が自然に存在し、これは従来のESNでは捉えにくい現象です。この制限を克服するために、Deep-ESNと呼ばれる新しい階層型リザーバーコンピューティングフレームワークを提案します。リザーバーを直接スタックするのではなく、ランダムに生成されたリザーバーと教師なしエンコーダーを組み合わせ、高次元の投影能力とリザーバーコンピューティングの効率的な学習を保持します。この複数の投影エンコードシステムを通じて、ESNの共線性問題を軽減するだけでなく、各層のマルチスケールダイナミクスも捉えます。Deep-ESN内の特徴リンクはマルチスケール情報融合を提供し、ネットワークの時系列適合能力を向上させます。また、Deep-ESNの安定性条件と計算複雑性の導出も提示しました。結果は、効率的な教師なしエンコーダー（例：PCA）を持つDeep-ESNが浅いESNと同様に効率的に学習できることを示しており、従来のリザーバーコンピューティングネットワークの主要な計算上の利点を保持しています。
---

In the experiments, we demonstrated empirically that our
Deep-ESNs outperform other baselines, including other approaches to multiscale ESNs, on four time series (two chaotic
systems and two real-world time series). Furthermore, we
found that increasing the size of the reservoirs generally improved performance, while increasing the size of the encoder
layer showed smaller improvements. We also showed that
increasing the depth of the network could either help or hurt
performance, depending on the problem. This demonstrates
that it is important to set the network structure parameters
using cross-validation.
We also evaluated how the model overcomes the collinearity
problem by measuring the condition numbers of the generated
representations at different layers of the network. We found
that using the encoders controlled this redundancy, especially
in the case of PCA. On the other hand, simply stacking
reservoirs as in MESM [23] leads to higher condition numbers
overall. This suggests that the encoders are a vital part of the
design of the system, and one of their main effects is to control
the collinearity in deeper reservoirs.
---
実験では、Deep-ESNが他のマルチスケールESNへのアプローチを含む他のベースラインを4つの時系列（2つのカオスシステムと2つの実世界の時系列）で上回ることを実証しました。さらに、リザーバーのサイズを大きくすることで一般的にパフォーマンスが向上する一方で、エンコーダ層のサイズを大きくしても改善は小さかったことがわかりました。また、ネットワークの深さを増やすことで、問題によってはパフォーマンスが向上したり低下したりすることも示しました。これは、ネットワーク構造パラメータをクロスバリデーションを使用して設定することが重要であることを示しています。
また、モデルがどのように共線性問題を克服するかを、ネットワークの異なる層で生成された表現の条件数を測定することで評価しました。エンコーダを使用することで、この冗長性が制御され、特にPCAの場合に顕著であることがわかりました。一方で、MESM [23]のようにリザーバーを単純にスタックすると、全体的に条件数が高くなることがわかりました。これは、エンコーダがシステムの設計において重要な部分であり、その主な効果の1つが深いリザーバーにおける共線性を制御することであることを示唆しています。
---

Finally, we investigated the multiscale dynamics in our models by using a perturbation analysis. We found that all of our
models demonstrated long-term memory for the perturbation,
which was most evident in the final layer. The MESM seemed
to never quite recover from the perturbation, with very small
but persistent effects. We also found the four time series we
used have different multiscale structures. Thus, the different
hierarchies of Deep-ESNs can deal with various multiscale
dynamics.
Reservoir computing is an efficient method to construct
recurrent networks that model dynamical systems. This is
in stark contrast to deep learning systems, which require extensive training. The former pursues conciseness and effectiveness, but the latter focuses on the capacity to learn abstract,
complex features in the service of the task. Thus, there is a gap
between the merits and weaknesses of these two approaches,
and a potentially fruitful future direction is to discover a way
to bridge these two models, and achieve a balance between
the efficiency of one and the feature learning of the other. Our
Deep-ESN is a first step towards bridging this gap between
reservoir computing and deep learning.
---
最後に、摂動分析を使用してモデル内のマルチスケールダイナミクスを調査しました。すべてのモデルが摂動に対して長期記憶を示し、これは最終層で最も顕著でした。MESMは摂動から完全には回復せず、非常に小さいが持続的な影響を与え続けるようでした。また、使用した4つの時系列が異なるマルチスケール構造を持つこともわかりました。したがって、Deep-ESNの異なる階層はさまざまなマルチスケールダイナミクスに対処できます。
リザーバーコンピューティングは、動的システムをモデル化するリカレントネットワークを構築する効率的な方法です。これは、広範なトレーニングを必要とする深層学習システムとは対照的です。前者は簡潔さと効果を追求しますが、後者はタスクのために抽象的で複雑な特徴を学習する能力に焦点を当てています。したがって、これら2つのアプローチの利点と弱点の間にはギャップがあり、
潜在的に有望な将来の方向性は、これら2つのモデルを橋渡しする方法を発見し、一方の効率ともう一方の特徴学習のバランスを達成することです。私たちのDeep-ESNは、このギャップを埋めるための第一歩です。
---

In order to study the memory properties of deep ESNs,
Gallicchio and Micheli [22] performed an empirical analysis of deep ESNs with leaky integrator units. They constructed an
artificial random time series out of 10 1-out-of-N inputs, and
constructed a second time series by adding a “typo” partway
through the series. The goal was to measure how long the
representation differed between the original series and the one
with a typo. They found that varying the integration parameter,
slowing the integration through the stacks lead to very long
memory. It is unclear how well this will generalize to realistic
time series, but it is an interesting observation.
---
深いESNの記憶特性を研究するために、GallicchioとMicheli [22]は、リーキーインテグレータユニットを持つ深いESNの経験的分析を行いました。彼らは10個の1-out-of-N入力から人工的なランダム時系列を構築し、シリーズの途中で「タイプミス」を追加することで2番目の時系列を構築しました。目標は、元のシリーズとタイプミスのあるシリーズとの間で表現がどれだけ長く異なるかを測定することでした。彼らは、統合パラメータを変えることで、スタックを通じて統合を遅くすることが非常に長い記憶につながることを発見しました。これが現実的な時系列にどれほど一般化されるかは不明ですが、興味深い観察です。
---

To retain the computational advantages of RC, the encoder
T should have low learning cost. Three dimensionality reduction (DR) techniques are used.
---
計算上の利点を保持するために、エンコーダTは低い学習コストを持つ必要があります。3つの次元削減（DR）技術が使用されます。
---

Principal Component Analysis (PCA) is a popular DR
statistical method. PCA adopts an orthogonal base transformation to project the observations into a linearly uncorrelated
low-dimensional representation where the selected orthogonal
bases are called principal components. In mathematical terms,
PCA attempts to find a linear mapping W ∈ R
D×M (M < D)
that maximizes the following optimization problem.
---
主成分分析（PCA）は、一般的な次元削減（DR）統計手法です。PCAは、観測値を線形的に相関のない低次元表現に投影するために直交基底変換を採用し、選択された直交基底は主成分と呼ばれます。数学的には、PCAは次の最適化問題を最大化する線形マッピングW ∈ R
D×M（M < D）を見つけようとします。
---

Reasonable setting of the hyperparameters is vital to building a high performance RC network. There are three commonly used strategies: direct method (based on user’s experience), grid search and heuristic optimization. The former
two strategies are used for general single-reservoir ESNs,
but they are unsuitable for Deep-ESNs due to its larger
parameter space. Thus, we adopt heuristic optimization to set
hyperparameters. The genetic algorithm (GA) is a commonlyused heuristic technique to generate high-quality solutions to
optimization and search problems [39]. The GA works on a
population of candidate solutions. First, the fitness of every
individual in the population is evaluated in each iteration
(called a “generation”), where the fitness is the value of the
objective function. The more fit individuals are stochastically
selected from the current population and used to form a new
generation by three biologically-inspired operators: mutation,
crossover and selection. Finally, the algorithm terminates when
a maximum number of generations or a satisfactory fitness
level is reached.
In our Deep-ESN, we view the cascaded hyper-parameter
vector of IS, SR and γ of each reservoir as an individual,
and the search space is constrained to the interval [0, 1].
Additionally, we use the prediction error of the system as the
fitness value of individual (the smaller loss, the higher fitness).
We set a population size to 40 individuals and evaluate 80
generations. In all the experiments that follow, we use the
training set to optimize the hyperparameters, with the fitness
measured on the validation set.
---
ハイパーパラメータの適切な設定は、高性能のRCネットワークを構築するために重要です。一般的に使用される3つの戦略があります：直接法（ユーザーの経験に基づく）、グリッドサーチ、およびヒューリスティック最適化。前者の2つの戦略は、一般的な単一リザーバーESNに使用されますが、Deep-ESNはパラメータ空間が大きいため不適切です。したがって、ヒューリスティック最適化を採用してハイパーパラメータを設定します。遺伝的アルゴリズム（GA）は、最適化と探索問題に高品質な解を生成するためによく使用されるヒューリスティック手法です[39]。GAは候補解の集団で動作します。まず、各世代（「世代」と呼ばれる）で集団内のすべての個体の適合度が評価されます。適合度は目的関数の値です。より適合した個体は現在の集団から確率的に選択され、突然変異、交叉、選択という3つの生物学的にインスパイアされた演算子によって新しい世代を形成します。最後に、最大世代数または満足できる適合度レベルに達するとアルゴリズムは終了します。
Deep-ESNでは、各リザーバーのIS、SR、およびγのカスケードハイパーパラメータベクトルを個体と見なし、探索空間は区間[0, 1]に制約されます。
さらに、システムの予測誤差を個体の適合度値として使用します（損失が小さいほど適合度が高い）。個体数を40に設定し、80世代を評価します。以下のすべての実験では、ハイパーパラメータを最適化するためにトレーニングセットを使用し、適合度は検証セットで測定されます。
---

In this section, we provide a comprehensive experimental
analysis of our proposed Deep-ESN on two chaotic systems
and two real world time series. Specifically, these time series
are 1) the Mackey-Glass system (MGS); 2) NARMA system;
3) the monthly sunspot series and 4) a daily minimumtemperature series. Fig. 3 shows examples of these time series.
Qualitatively, we see that the NARMA dataset and the daily
minimum-temperature series present strong nonlinearity, the
monthly sunspot series presents nonlinearity at its peaks, and
the MGS chaotic series is relatively smooth.
To evaluate the effectiveness of proposed Deep-ESN, we
compare with four baseline models, including a singlereservoir ESN with leaky neurons [28], the aforementioned
two-layer ESN variants: ϕ-ESN [25], R2SP [26], and the
most recent hierarchical ESN-based model called multilayered
echo-state machine (MESM) [23]. In fact, MESM can be
viewed as a simplified variant of the Deep-ESN without
encoders. Since the core of our work is exploring and analyzing the effectiveness of hierarchical schema, we ignore other
variants of ESNs, for example, the Simple Cycle Reservoir
(SCR) [40] with a circular topology reservoir, and the support
vector echo state machine (SVESM) [41] which optimizes the
output weights with an SVM. It is worth noting that these
single-reservoir variants (SCR, SVESM) can be viewed as a
module and integrated into our Deep-ESN framework.
We also compare the performance among Deep-ESNs with
various encoders (PCA, ELM-AE, RP). Furthermore, in these
comparisons, we conduct experiments on Deep-ESNs with
or without feature links to evaluate the impact of fusing
multiscale dynamics to the outputs.
---
このセクションでは、提案されたDeep-ESNの包括的な実験分析を2つのカオスシステムと2つの実世界の時系列で行います。具体的には、これらの時系列は1) マッキー・グラスシステム（MGS）、2) NARMAシステム、3) 月間日照スポットシリーズ、4) 日次最低気温シリーズです。図3はこれらの時系列の例を示しています。定性的には、NARMAデータセットと日次最低気温シリーズは強い非線形性を示し、月間日照スポットシリーズはピークで非線形性を示し、MGSカオスシリーズは比較的滑らかです。
提案されたDeep-ESNの効果を評価するために、4つのベースラインモデルと比較します。これには、リーキーニューロンを持つ単一リザーバーESN [28]、前述の2層ESNバリアント：ϕ-ESN [25]、R2SP [26]、および最新の階層型ESNベースのモデルである多層エコー状態マシン（MESM）[23]が含まれます。実際、MESMはエンコーダなしのDeep-ESNの簡略化されたバリアントと見なすことができます。私たちの研究の核心は階層スキーマの効果を探求し分析することなので、他のESNのバリアント、例えば、円形トポロジーリザーバーを持つ単純サイクルリザーバー（SCR）[40]や、SVMで出力重みを最適化するサポートベクターエコー状態マシン（SVESM）[41]などは無視します 。これらの単一リザーバーバリアント（SCR、SVESM）はモジュールとして見なすことができ、私たちのDeep-ESNフレームワークに統合できます。
また、さまざまなエンコーダ（PCA、ELM-AE、RP）を持つDeep-ESNのパフォーマンスを比較します。さらに、これらの比較では、マルチスケールダイナミクスを出力に融合する影響を評価するために、特徴リンクの有無にかかわらずDeep-ESNで実験を行います。
---

As seen in Fig.9, for all the methods, the largest redundancy
occurs in the first reservoir. With the multiple state transitions in its hierarchical direction, MESM does not reduce redundancy any further after two layers. A high condition number in
MESM lowers the accuracy of the linear regression. Compared
with MESM, our Deep-ESN works well with its encoders. We
can see that the higher the layer, the redundancy on reservoirs
will be less, especially with PCA, although this flattens out
after E2 or E3. In fact, after R4, the condition number appears
to increase for ELM and RP, which suggests why the PCA
encoder is more effective.
---
図9に示すように、すべての方法で最大の冗長性は最初のリザーバーで発生します。階層的な方向での複数の状態遷移により、MESMは2層以降冗長性をさらに減少させません。MESMの高い条件数は線形回帰の精度を低下させます。MESMと比較して、私たちのDeep-ESNはエンコーダともうまく機能します。層が高くなるほど、リザーバーの冗長性は減少し、特にPCAを使用した場合は顕著ですが、E2またはE3以降は平坦化します。実際、R4以降、ELMとRPでは条件数が増加するように見えます。これは、PCAエンコーダがより効果的である理由を示唆しています。
---

