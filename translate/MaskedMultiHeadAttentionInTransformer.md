# translate English sentence to Japanese
# only translate the sentence to Japanese after the "--" line
# and add a new line after the Japanese sentence

Transformers have become a powerful tool in tasks like language translation and text generation. While the encoder helps understand the input, the decoder is responsible for generating the output. Two key components of the decoder are multi-head attention and cross-attention, which allow it to focus on different parts of the input and the output simultaneously. In this blog, we’ll break down how these attention mechanisms work, preparing you for a deeper understanding of the transformer decoder in upcoming posts.

--

トランスフォーマーは、言語翻訳やテキスト生成などのタスクにおいて強力なツールとなっている。エンコーダーが入力の理解を助ける一方で、デコーダーは出力の生成を担う。デコーダーの2つの重要な構成要素は、入力と出力の異なる部分に同時に注目することを可能にする、マルチヘッドアテンションとクロスアテンションである。このブログでは、これらのアテンション・メカニズムがどのように働くかを説明し、今後の記事でトランスフォーマー・デコーダーをより深く理解するための準備をします。

-

When we talk about the transformer decoder, it’s important to know that it behaves differently during training and inference (prediction). At inference time, the decoder works in an auto-regressive manner, meaning it generates each token in a sequence by using the tokens generated so far. In contrast, during training, the decoder doesn’t rely on its own previous outputs but uses the actual target sequence, which allows it to be faster and more efficient.

--

トランスフォーマー・デコーダーについて話すとき、トレーニングと推論（予測）で異なる動作をすることを知っておくことが重要です。推論時、デコーダーは自己回帰的に動作し、これまでに生成されたトークンを使用してシーケンス内の各トークンを生成します。対照的に、トレーニング中は、デコーダーは自分の以前の出力に依存せず、実際のターゲットシーケンスを使用するため、より速く効率的になります。

-

So, what exactly is an auto-regressive model? In deep learning, auto-regressive models are those that generate data points in a sequence by conditioning each new data point on the previously generated ones. For example, let’s say you’re using a machine learning model to predict stock prices. To predict the price for Friday, the model will consider its predictions for Wednesday and Thursday, adjusting the prediction for Friday based on the prior information. Similarly, in a transformer decoder, each new token prediction depends on the tokens that were generated before it.

自己回帰モデルとは、深層学習において、生成された各新しいデータポイントを以前に生成されたものに条件付けてシーケンス内のデータポイントを生成するモデルのことです。たとえば、機械学習モデルを使用して株価を予測しているとしましょう。金曜日の価格を予測するために、水曜日と木曜日の予測を考慮し、以前の情報に基づいて金曜日の予測を調整します。同様に、トランスフォーマー・デコーダーでは、新しいトークンの予測は、その前に生成されたトークンに依存しています。

--

Now, you might be wondering why most sequence-to-sequence models, such as those used for tasks like language translation, text generation, or summarization, are auto-regressive. We can't just generate all the words at once, instead of doing it step by step.

これは、トランスフォーマー・デコーダーが自己回帰的である理由です。たとえば、言語翻訳やテキスト生成、要約などのタスクに使用されるシーケンスツーシーケンスモデルのほとんどは自己回帰的です。すべての単語を一度に生成するのではなく、一歩ずつ生成する必要があります。

-

The reason for this is simple: when generating text, the next word often depends on the words that came before it. In tasks like language translation or text generation, each word you generate informs the next word. This is why we can’t produce all the words simultaneously. It’s the sequential nature of the data that forces the model to generate outputs one at a time. Essentially, future words rely on past words, so generating them all at once would result in a loss of context and coherence. This is why models that handle such tasks need to be auto-regressive — there’s simply no other way to model sequential data effectively.

--

自己回帰的な理由は簡単です。テキストを生成する際、次の単語は前の単語に依存することが多いからです。言語翻訳やテキスト生成などのタスクでは、生成する各単語が次の単語に影響を与えます。したがって、すべての単語を同時に生成することはできません。データの逐次的な性質が、モデルに出力を1つずつ生成させることを強制します。本質的に、未来の単語は過去の単語に依存しているため、一度にすべてを生成すると文脈と一貫性が失われます。このため、そのようなタスクを処理するモデルは自己回帰的である必要があります。逐次データを効果的にモデル化する方法は他にありません。

-

The fun part here is that while the transformer decoder is auto-regressive during inference (when it’s predicting or generating outputs), it behaves differently during training. During training, the decoder doesn’t follow the auto-regressive approach, which may seem odd. Shouldn’t the decoder act the same way during both training and inference?

--

面白いのは、トランスフォーマー・デコーダーは推論中（予測または出力生成時）には自己回帰的である一方で、トレーニング中は異なる動作をすることです。トレーニング中、デコーダーは自己回帰的なアプローチに従わないため、奇妙に思えるかもしれません。トレーニングと推論の両方でデコーダーは同じように動作すべきではないでしょうか？

-

The key difference lies in something called masked self-attention, which we’ll dive into next. Understanding this concept will clarify why the transformer decoder behaves differently during training and inference, and how it helps the model efficiently learn and generate sequences.

--

マスクされた自己注意にある。この概念を理解することで、トランスフォーマー・デコーダーがトレーニングと推論で異なる動作をする理由と、モデルがシーケンスを効率的に学習し生成するのにどのように役立つかが明らかになります。

-

Let’s dive into the logic behind the statement that the transformer decoder behaves differently during inference compared to training. To prove this, we’ll first start by assuming that the decoder is auto-regressive during both inference and training.

--

トランスフォーマー・デコーダーが推論中とトレーニング中で異なる動作をするという主張の背後にある論理を掘り下げてみましょう。このことを証明するために、まずデコーダーが推論とトレーニングの両方で自己回帰的であると仮定します。

-

Consider a machine translation task, where we aim to build a deep learning model to translate English sentences to Hindi using a transformer. Suppose the model has been trained, and now it’s time to see how it performs during inference.

--

機械翻訳タスクを考えてみましょう。ここでは、トランスフォーマーを使用して英語の文をヒンディー語に翻訳する深層学習モデルを構築することを目指しています。モデルがトレーニングされたと仮定し、今、推論中のパフォーマンスを確認する時が来ました。

-

For example, let’s take the English sentence: “I am fine”. When this sentence is fed into the model during inference, the transformer’s encoder processes the entire input sentence in parallel, thanks to the self-attention mechanism. Each word — “I”, “am”, and “fine” — is processed simultaneously to generate respective embeddings. These embeddings undergo various layers such as multi-head attention, normalization, and feed-forward neural networks. After passing through these layers, we get final vectors for each word.

--

例えば、英語の文「I am fine」を考えてみましょう。この文が推論中にモデルに入力されると、トランスフォーマーのエンコーダーは自己注意メカニズムのおかげで、入力文全体を並行して処理します。「I」、「am」、「fine」の各単語は、それぞれの埋め込みを生成するために同時に処理されます。これらの埋め込みは、マルチヘッドアテンション、正規化、フィードフォワードニューラルネットワークなどのさまざまな層を経ます。これらの層を通過した後、各単語の最終ベクトルが得られます。

-

Now, these word vectors are sent to the transformer decoder. Here’s where the auto-regressive nature comes into play. The decoder generates the output one word at a time. It starts by predicting the first word based on a start token and the encoded input. Suppose the model predicts the word “मैं” (which means “I” in Hindi) as the first word.

--

この単語ベクトルはトランスフォーマー・デコーダーに送られます。ここで自己回帰的な性質が登場します。デコーダーは出力を1つずつ生成します。開始トークンとエンコードされた入力に基づいて最初の単語を予測することから始まります。モデルが最初の単語として「मैं」（ヒンディー語で「I」を意味する）を予測したとしましょう。

-

At this point, the decoder takes its first prediction, “मैं”, as an input along with the encoder’s output to predict the next word. This process continues word by word, with each subsequent word depending on the previously generated word.

--

この時点で、デコーダーは最初の予測「मैं」をエンコーダーの出力とともに入力として取り込み、次の単語を予測します。このプロセスは単語ごとに続き、各後続の単語は前に生成された単語に依存します。

-

However, mistakes can happen. Let’s assume at some point the model makes an incorrect prediction. Instead of predicting “बढ़िया” (which means “fine”), it predicts “घटिया” (which means “bad”). Due to the auto-regressive nature, this incorrect prediction is then fed into the next step, and the decoder continues with this error. By the end, the model outputs: “मैं घटिया हूं” (meaning “I am bad”) instead of “मैं बढ़िया हूं” (“I am fine”).

--

しかし、間違いが起こることがあります。ある時点で、モデルが誤った予測をすると仮定しましょう。「बढ़िया」（「fine」を意味する）ではなく、「घटिया」（「bad」を意味する）を予測します。自己回帰的な性質のため、この誤った予測は次のステップにフィードバックされ、デコーダーはこのエラーを続けます。最終的に、モデルは「मैं घटिया हूं」（「I am bad」を意味する）ではなく、「मैं बढ़िया हूं」（「I am fine」を意味する）を出力します。

-

This demonstrates how auto-regression works during inference. Each step relies on the previous output, making it crucial that earlier predictions are correct to avoid cascading errors.

let’s step back and dive into the training process of a transformer model. We’ve already discussed inference, but training follows a slightly different approach. To understand it better, let’s use the same example: translating the sentence “How are you?” from English to Hindi, where the translation would be “आप कैसे हैं”.

--

これにより、推論中の自己回帰の動作が示されます。各ステップは前の出力に依存しており、以前の予測が正確でないとエラーが連鎖する可能性があるため、重要です。
さて、トランスフォーマー・モデルのトレーニングプロセスに戻りましょう。推論についてはすでに説明しましたが、トレーニングは少し異なるアプローチを取ります。よりよく理解するために、同じ例を使いましょう。「How are you?」という文を英語からヒンディー語に翻訳する場合、翻訳は「आप कैसे हैं」となります。

-

When we train a transformer model, we start by passing our input sentence to the encoder. In this case, “How” goes into one position, “are” in the next, and “you” in the last position. The encoder processes these words and provides an output, which will be passed on to the decoder.

Now, here’s where the interesting part begins: auto-regressive training. Just like in inference, the decoder in training also works sequentially. Initially, we provide a start token (let’s say <start>) to signal the decoder to begin. Based on the input from the encoder and the start token, the decoder predicts the first word of the output sentence, which in this case should be "आप". However, if the model incorrectly predicts "तुम" instead, it’s no big deal at this point.

We don’t stop here. We move to the next step. But there’s a twist.

--

トランスフォーマーをトレーニングするとき、入力文をエンコーダーに渡すことから始めます。この場合、「How」は1つの位置に、「are」は次の位置に、「you」は最後の位置に入ります。エンコーダーはこれらの単語を処理し、出力を提供します。これはデコーダーに渡されます。
ここで面白い部分が始まります：自己回帰的なトレーニングです。推論と同様に、トレーニング中のデコーダーも逐次的に動作します。最初に開始トークン（<start>としましょう）を提供して、デコーダーに開始を知らせます。エンコーダーからの入力と開始トークンに基づいて、デコーダーは出力文の最初の単語を予測します。この場合、「आप」であるべきですが、モデルが「तुम」と誤って予測した場合、この時点では大した問題ではありません。
次のステップに進みます。しかし、ひねりがあります。

-

In the next time step, instead of feeding the decoder with the wrong prediction (“तुम”), we use the correct word from our dataset, “आप”. This technique is called teacher forcing. Even if the model makes a mistake, we feed the correct word in the next step during training.

The decoder continues, and we keep using the correct outputs at each time step. This ensures that the training process is smoother and doesn’t get derailed by incorrect predictions early on.

Now, why does this matter? Because teacher forcing allows us to train more efficiently.

--

次の時間ステップでは、デコーダーに誤った予測（「तुम」）を与える代わりに、データセットから正しい単語「आप」を使用します。この技術は教師強制と呼ばれます。モデルが間違いを犯しても、トレーニング中の次のステップで正しい単語を与えます。
デコーダーは続行し、各時間ステップで正しい出力を使用し続けます。これにより、トレーニングプロセスがスムーズになり、初期の誤った予測によって脱線することがありません。
このことはなぜ重要なのでしょうか？教師強制は、より効率的にトレーニングすることを可能にします。

-

If we were to treat the training process as fully auto-regressive, similar to inference, it would slow down the entire process. Think about it: every word in the sentence would have to be predicted one after the other. In the case of a small sentence like “How are you?”, we would need to pass through the decoder three times. But imagine a full paragraph with 300 words — the decoder would need to process each word individually, multiplying the time and computation required.

For a large dataset with thousands of rows, this sequential approach would make training painfully slow.

--

トレーニングプロセスを推論と同様に完全に自己回帰的に扱うと、全体のプロセスが遅くなります。考えてみてください：文中のすべての単語を1つずつ予測する必要があります。「How are you?」のような小さな文の場合、デコーダーを3回通過させる必要があります。しかし、300語の完全な段落を想像してみてください。デコーダーは各単語を個別に処理する必要があり、必要な時間と計算が増加します。
大規模なデータセットで何千もの行がある場合、この逐次的なアプローチはトレーニングを非常に遅くします。

-

Here’s the magic of teacher forcing: during training, we already know the correct output at each time step from the dataset. So, instead of predicting each word one by one, we can parallelize the entire process. The correct words from the dataset can be fed into the decoder all at once, rather than waiting for the previous step’s output. This eliminates the need for sequential execution in training.

In contrast, during inference, we don’t have the luxury of knowing the correct output beforehand, so the model must predict one token at a time. But in training, this constraint doesn’t exist, thanks to teacher forcing.

--

教師強制の魔法は、トレーニング中にデータセットから各時間ステップで正しい出力をすでに知っていることです。したがって、各単語を1つずつ予測する代わりに、全体のプロセスを並行化できます。データセットからの正しい単語を一度にデコーダーに与えることができ、前のステップの出力を待つ必要はありません。これにより、トレーニング中の逐次実行の必要が排除されます。
このように、トレーニング中は教師強制のおかげで、この制約は存在しません。



-

During inference, the prediction of each word depends on the previous word. For example, when predicting the second word, the model relies on the output of the first. But in training, this isn’t the case. The next word in the sequence is always available from the dataset, allowing us to skip the sequential nature.

Thus, we can execute multiple steps of the decoder simultaneously, significantly speeding up the training process. This is the core advantage of using teacher forcing.

--

トランスフォーマー・デコーダーは、推論中に各単語の予測が前の単語に依存します。たとえば、2番目の単語を予測する際、モデルは最初の出力に依存します。しかし、トレーニング中はそうではありません。シーケンス内の次の単語は常にデータセットから利用可能であり、逐次的な性質をスキップできます。
したがって、デコーダーの複数のステップを同時に実行でき、トレーニングプロセスを大幅に加速できます。これが教師強制を使用する主な利点です。

-

Now, here’s the fundamental problem: when you’re parallelizing, you’re sending all the words to the self-attention block simultaneously. While this works fine for generating contextual embeddings during training, there’s a serious issue: self-attention uses information from future tokens, which is unfair during inference.

Let’s break it down with a concrete example. At the start of our sentence, when we only have “आप”, the model should not have access to “कैसे” or “हैं” because they haven’t been generated yet. However, when training in parallel, self-attention does access this future information — using “कैसे” and “हैं” to influence the embedding of “आप”.

This is essentially cheating. During training, it’s possible to use future tokens because the entire sentence is available. But during inference (or real-world predictions), we don’t have access to future tokens. The model would have just predicted “आप”, but it doesn’t know what the next word will be. Therefore, using future tokens during training leads to a mismatch between training and inference behavior, creating a phenomenon known as data leakage.

--

ここで根本的な問題があります。並行化していると、すべての単語を同時に自己注意ブロックに送信しています。これはトレーニング中にコンテキスト埋め込みを生成するのには問題ありませんが、深刻な問題があります。自己注意は未来のトークンからの情報を使用しますが、推論中には不公平です。
具体的な例で説明しましょう。文の最初に「आप」しかないとき、モデルは「कैसे」や「हैं」にアクセスできるべきではありません。なぜなら、それらはまだ生成されていないからです。しかし、並行してトレーニングすると、自己注意はこの未来の情報にアクセスします。「कैसे」と「हैं」を使用して「आप」の埋め込みに影響を与えます。
これは本質的に不正行為です。トレーニング中は、文全体が利用可能なため、未来のトークンを使用することができます。しかし、推論（または実際の予測）中には、未来のトークンにアクセスできません。モデルは「आप」を予測したばかりですが、次の単語が何になるかはわかりません。したがって、トレーニング中に未来のトークンを使用すると、トレーニングと推論の動作に不一致が生じ、「データリーク」として知られる現象を引き起こします。

-

When your model accesses future information during training, it learns based on insights that won’t be available during inference. This results in a model that performs well on the training data (where future words are available) but struggles in real-world scenarios where it must predict each word step-by-step.

This is a classic example of data leakage — a situation where the model has access to extra information during training that it wouldn’t have during inference. Consequently, your model might perform exceedingly well on the training data but fail to generalize when making predictions on new, unseen data.

In autoregressive models, training is slow because tokens are processed one by one, but there’s no data leakage since future tokens aren’t visible. To speed things up, non-autoregressive models process tokens in parallel, but this causes data leakage, as current tokens can see future ones, which won’t happen during real predictions. This creates a dilemma: autoregressive models are accurate but slow, while non-autoregressive models are fast but prone to mistakes. The solution to this problem lies in revisiting the calculations in self-attention, where we can find a way to balance speed and accuracy.

--

トレーニング中にモデルが未来の情報にアクセスすると、推論中には利用できない洞察に基づいて学習します。これにより、トレーニングデータ（未来の単語が利用可能な場合）ではうまく機能するモデルが得られますが、実際のシナリオでは、各単語をステップバイステップで予測する必要があります。
これはデータリークの古典的な例です。トレーニング中にモデルが推論中には利用できない追加情報にアクセスする状況です。その結果、モデルはトレーニングデータでは非常に良好なパフォーマンスを示すかもしれませんが、新しい未確認のデータに対して予測を行う際には一般化に苦労します。
自己回帰モデルでは、トークンが1つずつ処理されるためトレーニングは遅いですが、未来のトークンは表示されないためデータリークはありません。トークンを並行して処理する非自己回帰モデルを使用すると、現在のトークンが未来のトークンを見ることができるため、データリークが発生します。これは実際の予測中には起こりません。このため、自己回帰モデルは正確ですが遅く、非自己回帰モデルは速いですが間違いを犯しやすいというジレンマが生じます。この問題の解決策は、自己注意の計算を再検討することにあります。ここで、速度と精度のバランスを取る方法を見つけることができます。

-

To solve the issue of balancing speed and accuracy in transformer training, we need to dive into the details of self-attention. The key lies in the way self-attention processes input sequences. Let’s take an example sentence like “आप कैसे हैं.” First, we calculate the embeddings for each word and pass them into the self-attention block. Within this block, each word’s embedding interacts with three matrices: the query (Q), key (K), and value (V) matrices. By performing dot products with these matrices, we generate three new vectors for each word — query, key, and value vectors.

Next, we stack the query vectors to form a query matrix, the key vectors to form a key matrix, and the value vectors to form a value matrix. This allows us to perform matrix operations efficiently.

The self-attention process begins by taking the dot product between the query and key matrices, resulting in a new matrix representing the attention scores. These scores are scaled and normalized using softmax to ensure they add up to one. Then, the attention scores are multiplied by the value matrix to compute the contextual embeddings for each word.

However, to prevent future tokens from influencing current tokens (data leakage), we use a technique called “masking.” We introduce a mask matrix where values corresponding to unwanted future token contributions are set to negative infinity.

--

トランスフォーマーのトレーニングにおける速度と精度のバランスを取る問題を解決するためには、自己注意の詳細に飛び込む必要があります。鍵となるのは、自己注意が入力シーケンスを処理する方法です。「आप कैसे हैं」のような例文を考えてみましょう。まず、各単語の埋め込みを計算し、それらを自己注意ブロックに渡します。このブロック内で、各単語の埋め込みは3つの行列（クエリ（Q）、キー（K）、値（V）行列）と相互作用します。これらの行列とのドット積を実行することで、各単語に対してクエリ、キー、および値ベクトルという3つの新しいベクトルが生成されます。
次に、クエリベクトルをスタックしてクエリ行列を形成し、キーと値のベクトルをスタックしてそれぞれキー行列と値行列を形成します。これにより、行列演算を効率的に実行できます。
自己注意プロセスは、クエリ行列とキー行列の間でドット積を取ることから始まり、新しい行列が生成され、注意スコアを表します。これらのスコアはスケーリングされ、softmaxを使用して正規化され、合計が1になるようにします。次に、注意スコアを値行列に掛けて、各単語のコンテキスト埋め込みを計算します。
しかし、未来のトークンが現在のトークンに影響を与えるのを防ぐために（データリーク）、マスキングと呼ばれる技術を使用します。不要な未来のトークンの寄与に対応する値を負の無限大に設定したマスク行列を導入します。
このマスク行列は、自己注意の計算において未来のトークンを無視するために使用されます。具体的には、マスク行列を注意スコア行列に掛けることで、未来のトークンに関連するスコアをゼロにします。これにより、現在のトークンは未来のトークンからの情報を受け取らず、データリークを防ぎます。

-

When applying the softmax function, these values become zero, effectively preventing any leakage from future tokens during training.

This masking process allows us to combine the benefits of both autoregressive and non-autoregressive models. We can train in parallel for efficiency, while ensuring that current tokens only attend to past tokens, thereby avoiding data leakage. This solution strikes a balance between faster training and maintaining the integrity of token prediction.

--

ソフトマックス関数を適用すると、これらの値はゼロになり、学習中の将来のトークンからの漏れを効果的に防ぐことができる。
このマスキング処理により、自己回帰モデルと非自己回帰モデルの両方の利点を組み合わせることができる。効率化のために並列学習ができる一方、現在のトークンは過去のトークンにしか関与しないようにすることで、データの漏れを防ぐことができる。このソリューションは、トレーニングの高速化とトークン予測の完全性の維持の間でバランスをとっている

-