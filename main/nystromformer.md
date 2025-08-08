# translate English sentence to Japanese
# only translate the sentence to Japanese after the "--" line
# and add a new line after the Japanese sentence

The starting point of our work is to reduce the computational
cost of self-attention in Transformers using the Nystrom¨
method, widely adopted for matrix approximation (Williams
and Seeger 2001; Drineas and Mahoney 2005; Wang and
Zhang 2013). Following (Wang and Zhang 2013), we describe a potential strategy and its challenges for using the
Nystrom method to approximate the softmax matrix in self- ¨
attention by sampling a subset of columns and rows.
---
私たちの研究の出発点は、トランスフォーマーにおける自己注意の計算コストを、行列近似に広く採用されているNystrom法を使用して削減することです（Williams and Seeger 2001; Drineas and Mahoney 2005; Wang and Zhang 2013）。(Wang and Zhang 2013)に従い、自己注意におけるソフトマックス行列を近似するためにNystrom法を使用するための潜在的な戦略とその課題について説明します。これは、列と行のサブセットをサンプリングすることによって行われます。
---

AS is designated to be our sample matrix by sampling m columns and rows from S.
---
ASは、Sからm列と行をサンプリングすることによって、サンプル行列として指定されます。
---

S can be approximated via the basic quadrature technique of the Nystrom method. It begins ¨
with the singular value decomposition (SVD) of the sample
matrix, AS = UΛVT
, where U, V ∈ R m×m are orthogonal
matrices, Λ ∈ R m×m is a diagonal matrix. Based on the outof-sample columns approximation (Wang and Zhang 2013),
the explicit Nystrom form of S can be reconstructed with m
columns and m rows from S,
---
Sは、Nystrom法の基本的な数値積分技術を使用して近似できます。これは、サンプル行列ASの特異値分解（SVD）から始まります。AS = UΛVT
、ここでU、V ∈ R m×mは直交行列、Λ ∈ R m×mは対角行列です。サンプル外列の近似に基づいて（Wang and Zhang 2013）、Sの明示的なNystrom形式は、Sからm列とm行を使用して再構築できます
---

 Here, (4) suggests that the
n × n matrix S can be reconstructed by sampling m rows
(AS, BS) and m columns (AS, FS) from S and finding the
Nystrom approximation ¨ Sˆ.
---
ここで、(4)は、n × n行列Sがm行（AS、BS）とm列（AS、FS）をSからサンプリングし、
Nystrom近似Sˆを見つけることによって再構築できることを示唆しています。
---

Nystrom approximation for softmax matrix. ¨ We briefly
discuss how to construct the out-of-sample approximation
for the softmax matrix in self-attention using the standard
Nystrom method.
---
ソフトマックス行列のNystrom近似。¨
自己注意におけるソフトマックス行列のサンプル外近似を標準のNystrom法を使用して構築する方法について簡単に説明します。
---

More details of the matrix
representation is available in the supplement.
---
行列表現の詳細は補足資料にあります。
---

Unfortunately, (4) and (6) require calculating all entries in QKT
due to the softmax function, even though the approximation
only needs to access a subset of the columns of S
---
不幸なことに、(4)と(6)はソフトマックス関数のためにQKTのすべてのエントリを計算する必要があります
が、近似はSの列のサブセットにのみアクセスする必要があります。
---

The problem arises due to the denominator within the rowwise softmax function. Specifically, computing an element
in S requires a summation of the exponential of all elements
in the same row of QKT
. Thus, calculating [
AS
FS
]
needs accessing the full QKT
, shown in Fig. 1, and directly applying
Nystrom approximation as in (4) is not attractive.
---
この問題は、行ごとのソフトマックス関数内の分母によって発生します。具体的には、Sの要素を計算するには、QKTの同じ行のすべての要素の指数の合計が必要です。
したがって、[AS
FS]を計算するには、図1に示すように、完全なQKTにアクセスする必要があり、(4)のようにNystrom近似を直接適用することは魅力的ではありません。
---

A key challenge of Nystrom approximation. The orange ¨
block on the left shows a n × m sub-matrix of S used by Nystrom¨
matrix approximation in (4). Computing the sub-matrix, however,
requires all entries in the n × n matrix before the softmax function
(QKT
). Therefore, a direct application of Nystrom approximation ¨
is problematic.
---
左のオレンジ色のブロックは、(4)でNystrom法の行列近似に使用されるSのn × mサブ行列を示しています。ただし、サブ行列を計算するには、ソフトマックス関数の前のn × n行列のすべてのエントリ（QKT）が必要です。したがって、Nystrom近似の直接的な適用は問題があります。
---

Linearized Self-Attention via Nystrom Method ¨
We now adapt the Nystrom method to approximately cal- ¨
culate the full softmax matrix S. The basic idea is to use
landmarks K˜ and Q˜ from key K and query Q to derive
an efficient Nystrom approximation without accessing the ¨
full QKT
. When the number of landmarks, m, is much
smaller than the sequence length n, our Nystrom approxima- ¨
tion scales linearly w.r.t. input sequence length in the sense
of both memory and time.
---
Nystrom法を使用した線形化された自己注意
¨
Nystrom法を適応させて、完全なソフトマックス行列Sを近似的に計算します。基本的なアイデアは、キーKとクエリQからランドマークK˜とQ˜を使用して、完全なQKTにアクセスせずに効率的なNystrom近似を導出することです。ランドマークの数mがシーケンス長nよりもはるかに小さい場合、Nystrom近似は、メモリと時間の両方の観点から、入力シーケンス長に対して線形にスケールします。
---

Following the Nystrom method, we also start with the ¨
SVD of a smaller matrix, AS, and apply the basic quadrature technique. But instead of subsampling the matrix after
the softmax operation – as one should do in principle – the
main modification is to select landmarks Q˜ from queries Q
and K˜ from keys K before softmax and then form a m × m
matrix AS by applying the softmax operation on the landmarks. We also form the matrices corresponding to the left
and right matrices in (4) using landmarks Q˜ and K˜ . This
provides a n × m matrix and m × n matrix respectively.
With these three n × m, m × m, m × n matrices we constructed, our Nystrom approximation of the ¨ n × n matrix S
involves the multiplication of three matrices as in (4).
---
Nystrom法に従い、より小さな行列ASのSVDから始め、基本的な数値積分技術を適用します。しかし、ソフトマックス操作の後に行列をサブサンプリングするのではなく
– 原則として行うべきことですが – 主な変更点は、ソフトマックスの前にクエリQからランドマークQ˜を選択し、
キーKからランドマークK˜を選択し、ランドマークにソフトマックス操作を適用してm × m行列ASを形成することです。また、
ランドマークQ˜とK˜を使用して、(4)の左行列と右行列に対応する行列を形成します。これにより、それぞれn × m行列とm × n行列が得られます。
これらの3つのn × m、m × m、m × n行列を構築することで、n × n行列SのNystrom近似は、(4)のように3つの行列の乗算を含みます。
---

In the description that follows, we first define the matrix
form of landmarks. Then, based on the landmarks matrix,
we form the three matrices needed for our approximation.
---
ランドマークの行列形式を最初に定義します。その後、ランドマーク行列に基づいて、近似に必要な3つの行列を形成します。
---

