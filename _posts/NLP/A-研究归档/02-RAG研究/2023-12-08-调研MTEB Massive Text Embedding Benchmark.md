# 调研 MTEB & BEIR 

> 问题：是否要做所有的任务，是否先做检索任务(需要参考BEIR)
>
> 目前先检索任务+分类任务

## 实验设置

模型：33个

语言：112种

指标：速度、内存占用

访问方式：调用、API调用

数据集：58个，8类嵌入任务

Bitext 挖掘、分类、聚类、对分类、重新排序、检索、STS 和摘要

### ① 任务划分

* **聚类**：给定一组句子或段落，目标是将它们分组到有意义的集群中。批量大小为 32 和 k 的小批量 k-means 模型等于不同标签的数量 (Pedregosa et al., 2011) 在嵌入文本上进行训练。该模型使用 **v-measure** 进行评分（Rosenberg 和 Hirschberg，2007）。Vmeasure 不依赖于集群标签，因此标签的排列不会影响分数。

  ![image-20231117140828151](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark\image-20231117140828151.png)

  * RedditClustering：199个子版块的标题聚类。25个分段的聚类，每个分段有10-50个类，每个类有100-1000个句子
  * RedditClusteringP2P：使用Reddit帖子中的可用数据为MTEB创建数据集[^11]。该任务包括根据标题+帖子的subreddit进行聚类拼接。它包含10个拆分，每个拆分有10到100个集群和1,000到100,000个帖子。
  * StackExchangeClustering：来自121个堆栈交换的标题聚类。聚类25个片段，每个片段有10-50个类，每个类有100-1000个句子。
  * StackExchangeClusteringP2P：使用StackExchange posts[^12]中的可用数据为MTEB创建数据集。该任务包括根据标题和帖子的子reddit进行聚类拼接。它包含10个拆分，每个拆分有10到100个集群和5,000到10,000个帖子。
  * TwentyNewsgroupsClustering[^13]：对给定文章标题的20个新闻组数据集进行聚类，目标是找到新闻组(总共20个)。包含10个拆分，每个拆分包含20个类，每个拆分包含1,000到10,000个标题。

* **分类**： A 训练集和测试集嵌入了提供的模型。训练集嵌入用于训练具有 100 个最大迭代次数的逻辑回归分类器，该分类器在测试集上进行评分。主要指标是平均精度的**准确度**，另外提供了 f1。

  ![image-20231117140804552](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark\image-20231117140804552.png)

  * AmazonCounterfactual：一组为反事实检测 **对分类?** 注释的亚马逊客户评论。对于每个评论，标签是“反事实”或“非反事实”。这是一个具有 4 种可用语言的多语言数据集。
  * AmazonPolarity：一组为极性分类注释的亚马逊客户评论。对于每个评论，标签要么是“正面”，要么是“负面”。
  * AmazonReviews：一组亚马逊评论，旨在帮助多语言文本分类的研究。对于每个评论，标签是评论在 0 到 4（1-5 星）之间的分数。这是一个具有 6 种可用语言的多语言数据集。
  * Banking77：由带有相应意图注释的在线银行查询组成的数据集。对于每个用户查询，标签是 77 个意图中的一个意图，例如“activ_my_card”、“apple_pay”、“bank_transfer”等。
  * Emotion：具有六种基本情绪的英语 Twitter 消息数据集：愤怒、恐惧、喜悦、爱、悲伤和惊讶。
  * Imdb：带有正面或负面标签的大型电影评论数据集。
  * MassiveIntent：一组带有相关意图注释的 Amazon Alexa 虚拟助手话语。对于每个用户话语，标签是 60 个意图之一，例如“play_music”、“alarm_set”等。这是一个具有 51 种可用语言的多语言数据集。
  * MassiveScenario：一组带有相关意图注释的 Amazon Alexa 虚拟助手话语。对于每个用户话语，标签是 60 个场景中的一个主题，例如“音乐”、“天气”等。这是一个具有 51 种可用语言的多语言数据集。
  * MTOPDomain / TOPIntent：MTOP (Li et al., 2020) 基准的多语言句子数据集。有关详细信息，请参阅他们的论文。
  * ToxicConversations：数据集来自Kaggle competition[^14]。收集来自民间评论平台的评论，以及评论是否有毒的注释。
  * TweetSentimentExtraction：数据集来自Kaggle competition[^15]。将tweet的情绪分类为中性、积极或消极。

* **对分类**：提供了一对文本输入，并且需要分配标签。标签通常是表示重复或释义对的二元变量。嵌入了这两个文本，它们的距离是通过各种指标计算的（余弦相似度、点积、欧几里得距离、曼哈顿距离）。使用最佳二进制阈值精度，计算平均精度 f1，精度和召回率。基于**余弦相似度的平均精度分数**是主要的指标。

  ![image-20231117140915162](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark\image-20231117140915162.png)

  * SprintDuplicateQuestions：来自Sprint社区的问题集合。目标是将一对句子分类为重复或不重复。
  * TwitterSemEval2015：来自SemEval 2015研讨会的对推文的释义。目标是将一对推文分类为释义或非释义。
  * TwitterURLCorpus：推文的释义对。目标是将一对推文分类为释义/非释义。

[^11]:(https://huggingface.co/datasets/sentence-transformers/reddit-title-body)
[^12]:(https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_body_jsonl)
[^13]:(https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
[^14]:(https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)
[^15]:(https://www.kaggle.com/competitions/tweet-sentiment-extraction)



* **Bitext 挖掘**：输入是来自两个**不同语言**的**两组句子**。对于第一组中的每个句子，需要找到第二组中的最佳匹配。匹配通常是翻译。提供的模型用于嵌入每个句子，最接近的对是通过余弦相似度找到的。**F1** 作为双文本挖掘的主要指标。还计算了准确性、精度和召回率。

  ![image-20231117141058098](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark\image-20231117141058098.png)

  * BUCC：BUCC 为英语、法语、俄语、德语和中文提供了大量句子（每个约 10-70k），以及相关的对注释。这里的注释对对应于一对翻译的句子，即一个句子及其在另一种语言的翻译。
  * Tatoeba：Tatoeba 为 112 种语言提供了一组句子（每个句子 1000 个句子），其中包含带有注释的相关对。每对都是一个句子及其在另一种语言中的翻译。

* **重新排序**：输入是查询和相关和不相关参考文本列表。目的是根据结果与查询的相关性对结果进行排名。该模型用于嵌入参考，然后使用余弦相似度与查询进行比较。每个查询对生成的排名进行评分，并在所有查询中取平均值。指标是平均 MRR@k 和 **MAP**，后者是主要指标。

  ![image-20231117140947611](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark\image-20231117140947611.png)

  * AskUbuntuDupQuestions：来自 AskUbuntu 的问题，带有手动注释，将成对的问题标记为相似或不相似。
  * MindSmall：用于新闻推荐研究的大规模英语数据集。给定新闻文章标题对新闻文章标题进行排名。这个想法是从您正在阅读的新闻中推荐其他新闻。
  * SciDocsRR：根据其标题对相关科学论文进行排名。
  * StackOverflowDupQuestions：针对带有Java, JavaScript和Python标签的问题的任务，将问题列为重复或不重复。

* **检索**：每个数据集由一个语料库、查询和每个查询到语料库中的相关文档的映射组成。目标是找到这些相关文档。提供的模型用于嵌入所有查询，并使用余弦相似度计算所有语料库文档和相似度分数。根据分数对每个查询的语料库文档进行排名后，计算 nDCG@k、MRR@k、MAP@k、precision@k 和 recall@k 的几个 k 值。**nDCG@10** 作为主要指标。MTEB 重用来自 BEIR 的数据集和评估（Thakur 等人，2021 年）。下图来自论文BEIR。

  ![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/84c03c7f91235dfea8c9919160d9c6ac_22_Table_8_-1595916118.png)

  * ArguAna, ClimateFEVER, CQADupstack, DBPedia, FEVER, FiQA2018, HotpotQA, MSMARCO, NFCorpus, NQ, Quora, SCIDOCS, SciFact, Touche2020, TRECCOVID：参考BEIR

* **语义文本相似度 (STS)** ：给定一个句子对，目的是确定它们的相似度。标签是连续分数，数字更高，表明句子更相似。提供的模型用于嵌入句子，并使用各种距离度量计算它们的相似度。使用 Pearson 和 Spearman 相关性以基本事实相似性对距离进行基准测试。基于**余弦相似度的 Spearman** 相关性作为主要指标（Reimers 等人，2016 年）。分值为0-5之间的连续值。

  ![image-20231117141002995](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark\image-20231117141002995.png)

  * STS12, STS13, STS14, STS15, STS16, STS17, STS22, STSBenchmark：原始STS基准，分数从0到5。选择的句子包括来自图片说明、新闻标题和用户论坛的文本。它们总共包含1,000到20,000个句子。STS12 - STS16 & STSBenchmark是单英语的，STS17和STS22包含跨语言的句子对，目的是评估不同语言中两个句子的相似度。STS17有11个语言对(韩语、阿拉伯语、英语、法语、德语、土耳其语、西班牙语、意大利语和荷兰语)，STS22有18个语言对(阿拉伯语、英语、法语、德语、土耳其语、西班牙语、波兰语、意大利语、俄语和汉语)。
  * BIOSSES：包含100个来自生物医学领域的句子对。
  * SICK-R：涉及构成知识的句子(SICK)包含大量的句子对(10万个)，这些句子对在词汇、句法和语义上都很丰富。

* **摘要**：提供了一组人工编写的和机器生成的摘要。目的是对机器摘要进行评分。提供的模型首先用于嵌入所有摘要。对于每个机器摘要嵌入，计算所有人类摘要嵌入的距离。保留最接近的分数（例如最高余弦相似度），并用作模型对单个机器生成的摘要的分数。计算了 Pearson 和 Spearman 相关性与机器生成的摘要的基本事实人工评估。与 STS 一样，基于**余弦相似度的 Spearman** 相关性作为主要指标（Reimers 等人，2016 年）。

  ![image-20231117141109979](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark\image-20231117141109979.png)

  * SummEval：由CNN或DailyMail上训练的最新摘要模型生成的摘要以及人工注释。



## 相关工作

### 2.1 基准构建相关工作

1.  【文本嵌入首选Benchmark】SemEval datasets (Agirre et al., 2012, 2013, 2014, 2015, 2016) 
2.  (Super)GLUE (Wang et al., 2018, 2019)
3.  Big-BENCH (Srivastava et al., 2022) 
4.  evaluation frameworks (Gao et al., 2021a)
5.  SentEval (Conneau and Kiela, 2018) 
6.  USEB (Wang et al., 2021)
7.  BEIR 基准 (Thakur et al., 2021) 
8.  MTEB 



SemEval：要求嵌入向量是“几何接近”的。并且单个SemEval数据集的表达能力有限。

SentEval：聚合了多个STS数据集。专注于在嵌入之上微调**分类器**。它缺乏检索或聚类等任务，其中嵌入是在没有额外分类器的情况下直接比较的。2018年提出的不适用于基于Transformer的文本嵌入。

USEB：主要由重新排序任务组成。它不涵盖检索或分类等任务。

BEIR 基准：评估零样本信息检索嵌入的标准。

MTEB ：将来自不同嵌入任务的数据集统一到一个通用、可访问的评估框架中。

$MTEB =SemEval (STS11 - STS22) + BEIR +来自各种任务的各种其他数据集$

### 2.2 Embedding Models

Glove (Pennington et al., 2014) 等文本嵌入模型缺乏上下文感知，因此通常被标记为**词嵌入模型**。简单地说，就是Glove这样的模型只能生成每个词的固定的向量表示，而不考虑词在不同的句子中的含义和用法。例如，词“bank”在“river bank”和“bank account”中的意思是不同的，但Glove会给它们相同的向量。而且，**Glove这样的模型不能直接生成句子的向量表示，而是需要对句子中的所有词的向量进行平均，这样会损失很多信息。**例如，句子“I love you”和“You love me”的平均向量是一样的，但它们的语义是不同的。

​		词嵌入模型：$1*hidden layer+1*pooler layer$

​		pooler layer是为了固定输出向量的长度

Transformers (Vaswani et al., 2017) 通过 self-attention 将上下文感知注入到语言模型中。

BERT (Devlin et al., 2018) 使用 Transformer 架构并执行大规模的自我监督预训练。

BERT产生的模型可以直接用于通过平均操作生成文本嵌入，**就像Glove一样，是指在一些任务中，可以不用对BERT进行微调，而是直接使用预训练的模型，对输入的文本进行编码，然后对每个词的向量进行平均，得到一个文本级别的向量表示**。BERT后面可以接一个池化层生成文本嵌入，但这里说的平均操作并不是指池化层，而是指**对每个词的向量进行简单的算术平均，得到一个文本级别的向量**。

在InferSent（Conneau等人，2017）的基础上，SBERT（Reimers和Gurevych，2019）证明了对转换器进行额外的微调以获得有竞争力的嵌入性能是有益的。最近的微调嵌入模型使用对比损失目标对正文本对和负文本对进行监督微调（Gao等人，2021b；王等人，2021；Ni等人，2021b；Muennighoff，2022）。



由于有大量可用的预训练转换器（Wolf等人，2020），至少有同样多的潜在文本嵌入模型有待探索。**这导致了对哪种模型为从业者的嵌入用例提供了最佳性能的困惑。**

我们在 MTEB 上对**词嵌入**和**Transformer-base模型**进行了基准测试，**量化了通常要慢得多的上下文感知模型所带来的收益**。



结论1：没有单一最好方案，不同模型适配不同任务。



不同模型的适用任务/不适用任务。



语义-文本相似性（STS）任务，该任务要求模型嵌入具有几何紧密嵌入的相似句子。



SentEval（不支持Transformers）专注于在嵌入之上微调分类器。它缺乏像检索或聚类这样的任务，在这些任务中，嵌入可以在没有额外分类器的情况下直接进行比较。

由于STS基准测试的不足，引入了USEB（Wang et al.，2021），主要由重新排序任务组成。因此，它不包括检索或分类等任务。

BEIR基准（Thakur et al.，2021）已成为零样本信息检索嵌入评估的标准。



## 需求：

1. 多样性：MTEB 旨在提供对嵌入模型在各种用例中的可用性的理解。该基准包含 8 个不同的任务，每个任务最多 15 个数据集。在 MTEB 的 58 个总数据集中，10 个是多语言的，涵盖 112 种不同的语言。包含句子级和段落级数据集，以对比短文本和长文本的性能。
2. 简单性：MTEB 提供了一个简单的 API，用于插入任何给定文本列表的模型可以为具有一致形状的每个列表项生成一个向量。这使得可以对一组不同的模型进行基准测试成为可能。
3. 可扩展性：现有任务的新数据集可以通过指定任务的单个文件和上传数据的 Hugging Face 数据集名称在 MTEB 中进行基准测试（Lhoest 等人，2021 年）。新任务需要实施任务界面来加载数据和评估器进行基准测试。我们通过拉取请求来欢迎来自社区的数据集、任务或度量贡献，以继续开发 MTEB。
4. 再现性：通过在数据集和软件级别进行版本化，我们的目标是使在 MTEB 中重现结果变得容易。与 MTEB 基准一起提供了与本文中提供的所有结果相对应的 JSON 文件

## 数据集

长度上分类：

Sentence to sentence (S2S)：包括 文本 语义相似度 (STS) 

Paragraph to paragraph (P2P)：包括 聚类任务（构建为了S2S和P2P两个版本）MTEB对输入长度没有限制，必要时由模型截断。几个集群任务被框定为S2S和P2P任务。前者只比较标题，后者包括标题和内容。例如，对于ArxivClustering，摘要被连接到P2P设置中的标题。

Sentence to paragraph (S2P)：包括 检索任务 这里的查询是一个句子，而文档是由多个句子组成的长段落。

![img](..\..\..\..\..\zoeChen119.github.io\assets\img\2023-12-08-调研MTEB Massive Text Embedding Benchmark/340b117a45e38e27318ff1c4e6ad1adb_3_Figure_2_-237605191.png)

工作2：每个数据集取100个样本，用每个模型生成embedding，计算句表征之间的余弦相似度，作为数据集和数据集之间相似度的表征。

图 2 展示了 56 个 MTEB 数据集的相似性。

有几个数据集依赖于相同的语料库，如 ClimateFEVER 和 FEVER，结果得分为 1。

同一数据集的 S2S 和 P2P 变体也趋于相似。

科学数据集，如 SciDocsRR、SciFact 和 ArxivClustering，即使来自不同的任务（本例中为重新排序、检索和聚类），相互之间也显示出很高的相似性。

## 一些结论

除了MSMARCO数据集，其他的都是在test集上进行评估，dev集的使用方式参考Thakur et al. (2021)。

按照自监督和监督方法对模型进行分类。

![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/59441eeba60e1fa37044260140167ebf_5_Figure_3_-1412672351.png)

**图3：MTEB 性能随型号大小而变化。**最小的 SGPT 变体性能低于类似大小的 GTR 和 ST5 变体。这可能是由于 SGPT 采用了只对偏置进行微调的方法，只有当模型规模增大、偏置参数数量增加时，这种方法才能赶上完全微调（Muennighoff，2022 年）。

* 自监督方法：

（a）：基于Transformer的 BERT（Devlin 等人，2018 年）使用自监督掩码和句子预测任务进行训练。通过取序列长度的平均值（平均池化），该模型可直接用于生成文本嵌入。SimCSE-Unsup（Gao 等人，2021b）使用 BERT 作为基础，并执行额外的自我监督训练： Komninos（Komninos 和 Manandhar，2016 年）和 Glove（Pennington 等人，2014 年）是两个直接将单词映射到向量的单词嵌入模型。因此，它们的嵌入缺乏上下文意识，但却能显著提高速度。

* 监督方法：

原始Transformer模型（Vaswani 等人，2017 年）由编码器和解码器网络组成。后续的Transformer器通常只训练编码器，如 BERT（Devlin 等人，2018 年）或解码器，如 GPT（Radford 等人，2019 年）。

（a） Transformer-编码器方法：GTR（Ni et al.，2021b）和ST5（Ni et al，2021a）基于T5模型的编码器部分（Raffel et al.，2020），仅在微调数据集上有所不同。

在额外的自我监督训练后，ST5对NLI进行对比微调以适应STS任务。同时，GTR对MSMARCO进行了微调，并专注于检索任务。MPNet和MiniLM对应于预训练的MPNet和MiniLM模型的微调嵌入模型，它们使用不同的数据集来针对任何嵌入用例。

（b）Transformer-解码器方法：SGPT BiEncoders使用加权均值池对小于 0.1% 的预训练参数进行对比微调。与 ST5 和 GTR 相似，SGPT-nli 模型面向 STS，而 SGPT-msmarco 模型面向检索。SGPT-msmarco 模型用不同的特殊标记嵌入检索查询和文档，以帮助模型区分它们的作用。对于非检索任务，我们使用其查询表示法。我们对基于 GPT-NeoX、GPTJ和 BLOOM的公开 SGPT 模型进行了基准测试。另外，cpt-text将预先训练好的 GPT 解码器通过一个两阶段的过程，使用最后一个标记池来提供解码器的嵌入。我们通过 OpenAI Embeddings API 对它们的模型进行了基准测试。

（c）非Transformer ：LASER是我们唯一的上下文感知非变换器模型，它依赖于 LSTM。与 LaBSE 类似，该模型也是在并行数据上进行训练，并侧重于 bitxt 挖掘应用。

![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/59441eeba60e1fa37044260140167ebf_4_Table_1_-1813929762.png)

1. 自监督和监督模型之间存在巨大差距。自监督的大语言模型可以缩小这个差距，但仍然需要有监督的微调才能获得有竞争力的嵌入。
2. 嵌入的性能与模型大小密切相关，如图3。
3. 大多数MTEB任务中表现好的模型是数十亿参数模型。

### ① 结论-分类任务

1. ST5模型综合最优，ST5-XXL的平均性能最高，高于非ST5模型中最佳模型Ada Similarity 3%。

### ② 结论-聚类任务

1. MPNet最优，与ST5-XXL相当。（其实MPNet比ST5-XXL小50倍，可能是因为MPNet已经在各种数据集上微调过了）
2. 聚类需要大量的嵌入之间的coherent distances。
3. 像 SimCSE-sup 或 SGPTnli 这样的模型只在单一数据集 NLI 上进行过微调，当遇到微调过程中未见过的主题时，可能会产生不一致的嵌入。
4.  SGPTmsmarco 和 Ada Search 端点的查询嵌入分别与 SGPT-nli 和 Ada Similarity 端点具有竞争力。
5. 虽然 OpenAI 文档建议在聚类用例中使用相似性嵌入6，但在某些情况下，检索查询嵌入可能是更好的选择。

### ③ 结论-对分类

1. GTR-XL和GTR-XXL具有最强的性能。配对分类在其框架上最接近STS，但模型在两个任务上的排名明显不同。

### ④ 结论-重排序

1. MPNet和MiniLM模型在重新排序任务中表现出色。（在SciDocsRR（Cohan et al.，2020a）上，它们的表现远好于更大的模型，这可能是因为SciDocsRR的部分内容包含在它们的训练数据中。）
2. 我们的实验规模和模型预训练规模使得控制数据污染具有挑战性。因此，我们在MTEB得分中忽略了MTEB数据集与模型训练数据集的重叠。只要对足够多的数据集进行平均，我们认为这些影响是微不足道的。

### ⑤ 结论-检索

1. SGPT-5.8B-msmarco是MTEB中BEIR子集以及完整BEIR基准上的最佳嵌入模型
2. 使用BLOOM的更大的7.1B SGPT模型（Scao et al.，2022）表现明显较弱，这可能是由于BLOOM的多语言性。
3. 面向STS的模型（SimCSE、ST5、SGPTnli）在检索任务中表现不佳。
4. 检索任务的独特之处在于有两种不同类型的文本：查询和文档（“不对称”），而其他任务只有一种类型的文本（“对称”）。

### ⑥ 结论-STS&摘要

1. 检索模型（GTR、SGPT-msmarco）在STS上的性能较差，而ST5-XXL的性能最高。这突出了该领域在检索（不对称）和相似性（对称）用例中分为独立嵌入模型的分歧（Muennighoff，2022）。

### ⑦ 速度和性能最佳平衡者-MPNet

![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/340b117a45e38e27318ff1c4e6ad1adb_6_Figure_4_-681862854.png)

不同嵌入模型的性能、速度和生成嵌入结果的大小（圆圈大小）。每个示例的嵌入大小从 1.2 kB（Glove / Komninos）到 16.4 kB（SGPT-5.8B）不等。在 STS15 上使用 1x Nvidia A100 80GB 和 CUDA 11.6 对速度进行了基准测试。

我们在图 4 中研究了模型的延迟-性能权衡。该图允许在模型选择过程中大量删除候选模型。它将模型选择减少到三个群组：

1. 最高速度
2. 最高性能
3. 速度和性能

### ⑧ 中文上，SGPT最优，MPNet最具性价比（分类、STS任务）![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/340b117a45e38e27318ff1c4e6ad1adb_7_Figure_5_1635353320.png)

图5：**MTEB 多语言性能。LaBSE 在 Bitext 挖掘方面占主导地位，而分类和 STS 结果则参差不齐。**SGPT-BLOOM-7B1-msmarco 在对 BLOOM 进行预训练的语言（如中文、法语和葡萄牙语）上往往表现出色。MTEB 包含 10 个多语言数据集，涉及咬文挖掘、分类和 STS 任务。我们在图 5 中研究了这些数据集的性能。表格结果见表 12、表 13 和表 14。

### ⑨

![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/340b117a45e38e27318ff1c4e6ad1adb_19_Figure_6_569165297.png)

图6：模型和（8大）任务之间的皮尔逊相关性。左图：同一结构的尺寸变体显示出高度相关性。右图 聚类和重排的性能相关性最强，而摘要和分类与其他任务的相关性较弱。

[图6展示了不同的文本嵌入模型在不同的嵌入任务上的皮尔逊相关性，即模型的性能与任务的难度之间的线性关系](https://arxiv.org/abs/2210.07316)[1](https://arxiv.org/abs/2210.07316)。

[图6的皮尔逊相关性是这样计算的：首先，对于每个嵌入任务，计算所有模型的平均性能，作为任务的难度指标](https://arxiv.org/abs/2210.07316)[1](https://arxiv.org/abs/2210.07316)[。然后，对于每个模型，计算它在所有嵌入任务上的性能，作为模型的能力指标](https://arxiv.org/abs/2210.07316)[1](https://arxiv.org/abs/2210.07316)[。最后，对于每个模型，计算它的能力指标与任务的难度指标之间的皮尔逊相关系数，作为模型和任务的皮尔逊相关性](https://arxiv.org/abs/2210.07316)[1](https://arxiv.org/abs/2210.07316)。

[图6的目的是展示不同的文本嵌入模型在不同的嵌入任务上的一致性，即模型是否能够在不同难度的任务上保持稳定的性能](https://arxiv.org/abs/2210.07316)[1](https://arxiv.org/abs/2210.07316)[。图6的结果表明，没有哪一种文本嵌入方法在所有任务上都占据优势，这意味着文本嵌入领域还没有找到一种通用的方法，能够在所有嵌入任务上提供最佳的结果](https://arxiv.org/abs/2210.07316)[1](https://arxiv.org/abs/2210.07316)[2](https://github.com/embeddings-benchmark/mteb)。

## 本Benchmark局限性：

1. 缺乏文档级数据：MTEB 涵盖多个文本长度（S2S、P2P、S2P），但仍然**缺少非常长的文档**。MTEB 中最长的数据集有几百个单词，较长的文本大小可能与检索等用例相关。
2. 任务不平衡：MTEB 中的任务有不同数量的数据集，**摘要仅包含单个数据集**。这意味着在所有数据集上计算的 MTEB 平均分数偏向于具有许多数据集的任务，特别是检索、分类和聚类。随着 MTEB 的增长，我们希望为当前代表性不足的任务添加更多数据集，例如摘要或配对分类。
3. 多语言：MTEB包含多语言分类、STS和文本挖掘数据集。然而，**检索和聚类只支持英语**。SGPT-BLOOM-7B1-msmarco面向多语言检索数据集，由于缺乏这些数据集，因此无法在MTEB中进行全面的基准测试。
4. 额外的方法：文本嵌入通常用作下游模型的输入特征，例如在我们的分类任务中。这可能涉及其他形式，特别是图像内容。我们只专注于自然语言应用，并将文本嵌入的广泛基准测试作为未来工作的其他模式的输入。









工作1：先找到当前最优模型

![img](https://pdf.cdn.readpaper.com/parsed/fetch_target/340b117a45e38e27318ff1c4e6ad1adb_4_Table_1_-1874723018.png)






