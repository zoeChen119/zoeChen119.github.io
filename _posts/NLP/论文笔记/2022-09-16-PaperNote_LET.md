---
title: Paper Note:语义学优化中文NLP任务
date: 2022-09-16 10:10:56 +/-0800
categories: [NLP,论文笔记]
tags: [nlp,语言学,论文笔记]     # TAG names should always be lowercase
---
# Paper Note:语义学优化中文NLP任务

《LET: Linguistic Knowledge Enhanced Graph Transformer for Chinese Short Text Matching》  
作者：Boer Lyu, Lu Chen*, Su Zhu, Kai Yu  
上海交通大学 X-LANCE Lab || 北京媒体融合生产技术国家重点实验室  
会议：AAAI-21

创新点：引入语言学知识来对模型进行增强
目标模型：图Transformer  
目标任务：中文短文本匹配

摘要：  
目前的方法通常采用中文字符或词组作为input tokens。这样做有两个局限：1）一些中文词组“一词多义”，并且语义信息没有得到充分利用；2）由于“分词”这个老大难导致很多模型本身就有很多潜在问题。  
<font color=DeepSkyBlue> 本文引入HowNet作为外部知识库（external knowledge base），提出模型LET用来解决“词语歧义”的问题。</font>  
<font color=LightSkyBlue> 另外，本文采用the word lattice graph（词格图）作为input来保持多粒度的信息。</font>
LET模型也是预训练语言模型的“互补”模型。  

实验结果：在2个中文数据集上，LET优于各种经典的文本匹配方法。  
消融实验结果：语义信息和多粒度信息对文本匹配建模具有重要意义。  

![](/assets/img/2022-09-16-PaperNote_LET/2022-09-27-16-24-17.png)

## <center><font color=Tomato> Question </font></center>

1. 消融实验怎么做的？语义信息和多粒度信息的意义如何证明？  
2. the word lattice graph（词格图）是什么？还有什么方法能够表征“多粒度信息”？
   
> 消融实验就是控制变量法。

## <center><font color=LightSeaGreen> Challenge </font></center>
1. word ambiguity词语歧义
2. The correct division of words正确的词语划分

## Research Background 

### 1.词语歧义

> 处理歧义的现象称为“消歧”。

《中文多义词词典》

多义词举例：
> 包袱 bāofu
> (1) [cloth-wrapper]∶包裹物件用的布面
> (2) [a bundle wrapped in a cloth-wrapper]∶外包有布的包裹
> (3) [load]∶喻指精神上的负担
> (4) [burden]∶比喻某种负担,即使人沮丧、压抑或引起忧虑的事物丢掉包袱
> (5) [laughingstock]∶曲艺节目的笑料

> watch
> (1) v.看;注视;观看;观察;(短时间)照看，看护，照管;小心;当心;留意
> (2) n.表;手表;(旧时的)怀表;注意;注视;监视;观察;值班(人);警戒(人);守夜(人)

**词义的确定要考虑语境、句法和词语之间的关系。**

可见中文并没有明确的“词性”分别，而英文中有明确的词性。同样的，中文也没有明确的句法结构，而英文有。
1. 由于英文中的词语有明确的词性，所以在面对“一词多义”问题时，我们可以考虑**引入词性标识**,在中文中，这样做并不合适。
2. 由于英文中的句子有明确的句法结构，所以在面对“一词多义”问题时，我们可以考虑**识别谓词和当前词的语义角色**,在中文中，这样做也不合适。除非统计较为完善的每个中文词语常见的语义角色。或者是“依存句法分析”。
3. 因此，中文的词义确定**只能依靠语境，词语之间的关系**。

中文词语歧义的问题是由于汉语的“一词多义”现象导致的，一个词在不同的语境下表示不同的词义，**之前的研究**例如n-gram，skip-gram&cbow等认为从当前词的前后n个词可以表示当前词的上下文信息；而隐马模型则利用统计学基础认为从前面的所有词的含义累计传递来表示当前词的上下文信息。这统称为*基于词向量化*。这个方向的出发点是<u>从上下文语境信息入手确定词义信息</u>。
**另一个研究方向**是*基于知识图*，一个词可以包含多种含义（多义词）。每一个意义都在一个**概念**中表达，与其他**概念**联系起来，形成一个知识图。请注意，这里的<u>**概念**</u>在*义原*中也有出现。[^NLP中的消岐]

一种结合上下文语境信息和义项信息的词语消岐架构[^NLP词义消歧方法论]：
![](/assets/img/2022-09-16-PaperNote_LET/2022-09-27-16-45-54.png)


## Reference:
[^NLP中的消岐]:https://blog.csdn.net/fengdu78/article/details/118774187
[^NLP词义消歧方法论]:https://blog.csdn.net/xingzhe1993/article/details/107936754