---
title: Paper Note:词格法
date: 2022-10-08 14:10:56 +/-0800
categories: [NLP,论文笔记]
tags: [nlp,语言学,论文笔记]     # TAG names should always be lowercase
---
# Paper Note:词格法

《Lattice-BERT: Leveraging Multi-Granularity Representations in Chinese Pre-trained Language Models》  
作者：Yuxuan Lai, Yijia Liu, Yansong Feng, Songfang Huang, Dongyan Zhao
阿里巴巴达摩院 | 中国北京大学教育部计算语言学重点实验室
会议：

创新点：在中文预训练语言模型中的语言表示部分使用“多粒度表示”
目标模型：预训练语言模型
目标任务：语义表征

摘要：中文预训练的语言模型通常将文本作为字符序列来处理，而忽略了更粗略的颗粒度，例如单词。在这项工作中，我们提出了一种新的中文预训练范式--格子图，它明确地将词的表征与字符结合在一起，因此可以以多粒度的方式对一个句子建模。具体来说，我们从句子中的字和词构建一个格子图，并将所有这些文本单元送入转化器。我们设计了一个格子位置关注机制，以利用自我关注层中的格子结构。我们进一步提出了一个屏蔽段预测任务，以推动模型从格子中固有的丰富但冗余的信息中学习，同时避免学习意外的技巧。

实验结果：在11个中文自然语言理解任务上的实验表明，我们的模型在12层设置下可以带来1.5%的平均增长，这在CLUE基准上的基础大小的模型中达到了新的先进水平。进一步的分析表明，Lattice-BERT可以利用格子结构，其改进来自于对冗余信息和多角化表示的探索。

1.许多汉语单词的意思并不能通过直接组合其汉字的意思来完全理解

2.具体来说，我们将句子中的字符和单词组织为单词晶格（见图1），这使得模型能够从所有可能的分词结果中探索单词。
![](/assets/img/2022-10-08-PaperNote_词格法/2022-10-08-14-16-16.png)

## <center><font color=Tomato> Question </font></center>
1. 怎么整理成预训练模型的input？
2. 如何让模型理解词格之间的绝对位置和相对位置关系？
3. MLM任务不能用了，怎么办？

## <center><font color=LightSeaGreen> Challenge </font></center>
首先，BERT的原始输入是按位置排序的字符序列，这使得它很难消耗单词格子并保留多格子单元之间的位置关系。
其次，传统的掩蔽语言建模（MLM）任务可能会使基于词格的plm学习到意想不到的技巧。原因是这样的字格自然会引入冗余，即一个字符可以包含在多个文本单元中。
在MLM中，模型可能会参考与随机屏蔽的文本单元重叠的其他文本单元，而不是真实的上下文，这带来了信息泄露。

**解决方法**： 为了应对这些挑战，我们提出了一个基于格子的双向编码器表示法（Lattice-BERT）。
具体来说，我们设计了一个**格子位置注意力（LPA）**，以帮助变换器直接利用格子中文本单元之间的位置关系和距离。
此外，我们还提出了一个**屏蔽段预测（MSP）任务**，以避免语言建模中重叠的文本单元之间的潜在泄漏。
有了LPA和MSP，Lattice-BERT可以利用格子中的多粒度结构，从而直接利用格子结构来聚合粗粒度的单词信息，使各种下游任务受益。

![](/assets/img/2022-10-08-PaperNote_词格法/2022-10-08-14-36-56.png)

除了MSP，我们还使用Lan等人（2020）的句子顺序预测（SOP）任务对模型进行预训练，该模型预测两个连续的句子是否在输入中交换。

## 词格构建：
我们在由102K词汇组成的 基于由102000个词汇组成的格子 高频开放领域的词汇。所有出现在词汇表中的输入序列的子串被认为是输入的格子标记。使用<u>Aho-Corasick自动机（Aho and Corasick, 1975）</u>，这个构建过程可以在与语料库和词汇量的线性时间内完成。为了处理子串无意义的英语单词和数字，我们对那些词汇外的非中文输入使用字符序列，并保留词汇内的单词和词块。

我们根据词汇表**使用所有可能的词来构建词格，而不是采用更复杂的词格构建策略**。之前关于格子构建的研究工作（Lai等人，2019；Chen等人，2020；Li等人，2020b）表明，使用所有可能的词通常会产生更好的性能。我们认为<u>过度设计的格子构造方法</u>可能会使我们的模型在某些类型的文本上出现偏差，而且很可能会损害泛化效果。因此，在我们的案例中，我们让模型自己学习，以过滤在大规模语料库的预训练中使用所有可能的词所带来的噪音。

## 预训练细节
为了与之前的预训练工作相比较，我们实现了基础大小的模型，其中包含12层，768维的隐藏大小，以及12个注意力头。为了证明格子如何在较浅的架构中获得收益，并提供轻量级的基线，我们还进行了6层、8个注意头和512个隐藏尺寸的轻型模型。
**为了避免大词汇量在嵌入矩阵中引入过多的参数**，我们采用了Lan等人（2020，ALBERT）的嵌入分解技巧。因此，Lattice-BERT的参数基数为100M，仅比其字符级对应的参数（90M）多11%，并且小于RoBERTa-base（Liu等人，2019）（102M）和AMBERT（Zhang和Li，2020）（176M）。
<font color="DeepSkyBlue">网格位置关注中的位置关系和距离的建模只引入了12K的参数。</font>
> 这个是怎么量化的？

在BERT模型的预训练阶段，我们使用了一系列的中文文本，包括中文维基百科、知乎和网络新闻。我们的无标签数据中的字符总数为18.3G。我们遵循Liu等人（2019）的做法，用8K实例的大批次规模训练PLMs，共100K步。超参数和细节在附录C中给出。

我们详细介绍了Lattice-BERT在11个中文NLU任务上的微调结果。回答以下问题：
(1) Lattice-BERT是否比单粒度PLM和其他多粒度PLM表现得更好？
(2) 所提出的格子位置注意和遮蔽段预测对下游任务的贡献如何？
(3) Lattice-BERT是如何优于原来的字符级PLMs的？

我们用这些不同的下游任务对我们提出的Lattice-BERT模型进行了彻底的探测。每个任务的统计数据和超参数在附录B中详细说明。对于MSR和MSRA-NER，我们<u>用最佳的学习率设置运行了五次，并报告了平均分数，以确保结果的可靠性</u>。

## 消融实验
我们进行了消融实验，以调查我们提出的格子位置注意（LPA）和掩蔽段预测（MSP）在下游任务中的有效性。
* 序列长度：为了减少计算成本，我们将预训练设置在<u>序列长度为128个字符的Lite-size</u>上。
* 任务选择：我们从每个任务集群中选择一个任务。
* 评价指标：我们使用实体级的F1-score来突出对边界预测的影响，我们报告了5次运行的平均分数。
* 数据集选择：并使用CLUE任务的开发集。
  
![](/assets/img/2022-10-08-PaperNote_词格法/2022-10-08-15-22-29.png)

### 1. 消融实验检测MSP任务的效果：
![](/assets/img/2022-10-08-PaperNote_词格法/2022-10-08-15-21-11.png)

我们在表2中可以看到，任何一个模块（-Dis.-Rel. & -MSP）的消减都会导致平均分数的大幅下降。
1. 特别是，用vanilla MLM代替MSP，MSP的平均得分下降了1.6%。
2. 在WSC.任务中，需要长距离的依赖关系来解决核心词，差距高达3.1%。
3. 我们将这一下降追溯到预训练过程中，并观察到开发集（dev-data）上-MSP设置的MLM准确性为88.3%。
4. 然而，如果我们掩盖段内的标记并避免潜在的泄漏，准确率急剧下降到48.8%，远远低于使用MSP的LBERT训练的性能（56.6%）。**这一差距提供了证据，表明MSP任务阻止了PLM通过偷看一个语段中的重叠文本单元来欺骗目标，从而鼓励PLM对长距离的依赖性进行定性。**

### 2. 消融实验检测LPA格子注意力的效果：
1. 对于LPA方法，如果没有位置关系（-Rel.），实体级的F1得分在NER上下降了0.4%，在CMRC上的表现下降了0.7%。
2. 性能的下降与没有距离信息（-Dis.）的情况类似。
3. 如果没有其中任何一个（-Dis.-Rel.），差距分别扩大到0.5%和2.8%。
4. **NER和CMRC中的边界预测对局部语言结构如嵌套词或重叠的歧义更为敏感。**
5. 由于注意到了位置关系和距离特征，LBERT可以准确地模拟不同分割结果中的嵌套和重叠标记之间的互动。
6. 同时，如果没有距离信息，WSC的准确性明显下降。
7. 当代词和候选短语之间的字符数大于30，或在20到30之间时，性能分别下降了7.5%和5.8%。对于其他情况，下降幅度仅为0.4%。
通过对距离的明确建模，LBERT更准确地预测了长距离的核心推理关系。
8. 平均来说，如果没有LPA中的位置关系和距离建模，在三个任务上的性能下降了2.0%，这表明LPA在协助PLM利用词格中的多格结构方面的重要性。

### 3. LBERT如何改进细粒度的plm？
我们比较了LBERT和字符级BERT的预测结果-我们在开发集上的基础大小，以研究LBERT如何优于普通的细粒度plm。直观地说，格中的字级标记提供了粗粒度的语义，其参数是特征级输入。

## 一些发现

1. LBERT在较短的实例中带来了更多的改进
> 我们观察到，在TNEWS这一短文分类任务中，LBERT在较短的实例中带来了更多的改进，因为这些实例中的语句可能太短，无法为预测提供足够的背景。根据句子长度将开发集分为五个大小相等的仓，LBERT在最短和第二短的仓中分别比BERT-our好2.3%和1.3%，比其他实例的平均增益（0.6%）大。我们认为，词格中的冗余标记为这些短语句的语义提供了丰富的背景。例如，对于我们村的电影院/我们村的电影院这个短标题，由于在网格中引入了电影/电影、影院/电影院和电影院/电影院这些冗余词，LBERT将该实例归类为娱乐新闻而不是新闻故事。
2. LBERT擅于“预测候选词是否是某个段落的关键词”，原理是：通过利用格子中的冗余表达，从不同方面理解关键词
>  另一个案例是CSL任务，其目标是预测候选词是否是某个段落的关键词。对于那些LBERT平均从每个候选词中识别出两个以上的词级标记的情况，占到了数据集的47%，其性能增益为3.0%，明显大于其余部分的平均改进，即1.0%。我们认为LBERT通过利用格子中的冗余表达，从不同方面理解关键词。例如，从候选关键词太阳能电 池/太阳能电池中，太阳/太阳能，电池/电池，以及太阳能电池/太阳能电池都是格子标记。有了这些词级标记，LBERT可以将这个候选人与该段中的表达方式相匹配，如阳极/正极，光/光，电子/电子，离子/离子等。
3. LBERT减少了识别具有嵌套结构的实体的错误。
>  另一方面，对于MSRA-NER，LBERT减少了识别具有嵌套结构的实体的错误。平均而言，在LBERT中，预测的实体与黄金实体嵌套的错误案例数量减少了25%。例如，组织实体解放巴勒斯坦运动/巴勒斯坦民族解放运动与地点实体巴勒斯坦/巴勒斯坦嵌套在一起，并以一个指向组织的指标运 动/运动结束。字符级基线模型错误地将巴勒斯坦/巴勒斯坦和运/动分别识别为一个地点和一个组织。而LBERT在整合了解放/解放、巴勒斯坦/巴勒斯坦和运动/运动这些词之后，正确地识别了这个实体。通过预先训练的多粒度表征，LBERT同时融合了单词和字符的上下文信息，并成功地检测出了正确的实体。

## 一些思考
1. 案例研究
   
![](/assets/img/2022-10-08-PaperNote_词格法/2022-10-08-15-57-25.png)
案例研究。表3显示了CMRC中的一个例子，这是一个跨度选择MRC任务，其中模型从给定的文档中选择一个文本跨度来回答这个问题。在这种情况下，这个问题要求的是一款受其主题曲限制的游戏。BERTour错误地输出了主题曲《中国之歌》，因为在文档中没有明确的与游戏相关的表达。然而，LBERT找到了正确的答案，即剑仙传说v。一个可能的原因是，剑仙传说是格点结构词汇中的一个条目。LBERT可能是在训练前的语境中学习了一个著名电子游戏的实体，明确地利用它作为一个整体的表现。在训练前使用粗粒度文本单元，LBERT直接编码有关这些单元的知识，以有利于下游任务。

2. LBERT如何利用多粒度表示？

LBERT同时消耗输入序列中的所有单词和字符，但是该模型如何在训练前和下游任务中利用这种多粒度表示呢？**为了研究这一点，我们使用每个格点标记在所有层和所有头部中接收到的平均注意分数来表示其重要性。**
![](/assets/img/2022-10-08-PaperNote_词格法/2022-10-08-15-52-47.png)
如图4所示的例子，**在微调之前**，LBERT关注的是包括活/活，充实/充实，研究/研究，研究生/研究生，研究/调查等的标记。**在对具体任务进行微调之前，该模型捕获了句子的各个方面。在用MSRA-NER进行微调后，最集中的词成为**充实/充实，很/很，生活/生活，和研究/研究，即**黄金分割结果中的标记**，"研究|生 活|很|充实"，这对NER任务来说是很直观的。**错误的分割词 "研究生 "的注意力得分明显下降。**

另一方面，在对新闻标题分类任务进行微调后，TNEWS、LBERT倾向于关注充实/充实、研究生/研究生、生活/生活等内容。虽然这些标记不能在一个中文分词结果中共存，但**LBERT仍然可以利用来自各种可信分割的冗余信息来识别输入的主题**。这些结果表明，**Lattice-bert可以通过根据特定的下游任务将注意力转移到多粒度表示之间的不同方面来很好地管理晶格输入。**

3. 如何证明纳入格子而不是额外计算所带来的收益？
   
为了公平比较，我们确保LBERT和字符级基线（即BERT-our）在训练步骤相同的情况下，按照以前的工作（Diao等人，2020；Zhang和Li，2020）有相同的训练周期。因此，与BERT-our相比，在LBERT的预训练实例中多引入了35%的文本单元，与BERT-our相比，多引入了48%的计算资源来处理额外的词级标记（见附录C）。**为了说明纳入格子而不是额外计算所带来的收益，我们研究了在预训练中输入序列较长的序列大小的BERT-our，它的计算成本与LBERT相同。**我们发现在CLUE分类任务上，LBERT仍然比BERT-our平均多出2.2%。更多的细节将在附录D中阐述。
