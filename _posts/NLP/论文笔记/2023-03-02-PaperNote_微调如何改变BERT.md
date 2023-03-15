---
title: A Closer Look at How Fine-tuning Changes BERT
categories: [NLP,论文笔记]
tags: [nlp,预训练模型,论文笔记]     # TAG names should always be lowercase
---

# A Closer Look at How Fine-tuning Changes BERT

## 1 Introduction

##### 本文探究了这三个问题：
1. Does fine-tuning always improve performance?微调是否的确是总是能提高性能？
2. How does fine-tuning alter the representation to adjust for downstream tasks? 微调是怎么修正representation以适应下游任务的？
3. How does fine-tuning change the geometric structure of different layers?
微调具体是怎么改变不同层的几何结构的？

> 我的问题：微调怎么调整模型参数的？

##### 本文用 [2个probing-tech] 来检测BERT不同变体模型在 [5个下游任务] 上，representation的情况的：

2个probing tech

* classifier-based probing [^1] [^2]
* DIRECTPROBE [^3]

5个下游任务
* 词性标注
* <font color=OrangeRed>dependency head prediction依赖头检测</font>
* <font color=OrangeRed>preposition supersense role</font>
* function prediction函数预测
* 文本分类

##### previous发现
微调改变较高层而不是较低层，并且语言信息在微调过程中不会丢失。[^4] [^5]


##### 本文的新发现
1. 
   a. 微调引入了训练集和测试集之间的分歧，在大多数情况下，这不会损害泛化性。
   b. 有一种情况，微调会损害泛化性。这种情况时，经过微调后，训练集和测试集之间的差异（分歧）也是最大的。

2. 微调如何改变the representation space的<font color=OrangeRed>labeled region</font>。
   a. 对于<font color=OrangeRed>任务标签 不可线性分离</font>的表征，发现微调通过将具有相同标签的点归入少量的群组（最好是一个）来进行调整，从而简化了基础表征。
   b. 这样做使使用微调表示的标签比未微调untuned表示的标签更容易线性分离。
   c. 对于任务标签已经是线性可分离的表示，我们发现微调会将表示不同标签的点簇彼此推开，从而在标签之间引入较大的分离区域separating regions。簇不是简单地缩放点，而是以不同的方向和不同的范围移动（通过欧几里德距离测量）。
   d. 总的来说，与untuned的表示相比，这些簇变得遥远。我们推测，组之间的扩大区域允许一组更大的分类器，可以将它们分开，从而导致更好的泛化（§4.3）。
   本文通过研究跨任务微调的效果来验证这个“距离假设”。
   观察到，通过改变代表不同标签的簇之间的距离，相关任务的微调也可以为目标任务提供有用的信号（§4.4）。

3. 微调不会随意地改变the higher layers。
4. 微调在很大程度上保留了标签簇的相对位置，同时重新配置空间以适应下游任务。
5. Informally，可以说微调只是“slightly”改变了the higher layers。

## 2 [2个probing tech]

> 1. classifier-based probing [^1] [^2]
基于分类器的探针，用于评估在一个任务中，representation对分类器的support程度。
> 2. DIRECTPROBE [^3]
用于分析representation的几何结构


### (1) classifier-based probing [^1] [^2]

研究中，“经过训练的分类器”是最常用的probes。具体来说，如果想要了解，在一个task中，一个representation对labels的编码程度，那么就在它上面训练一个probing classifier（训练的时候，embeddings本身保持冻结）。

在本文的研究中，用的是一个2-layers的神经网络作为probe classifiers。
其他细节：
* 用网格搜索选择the best 超参数；
* 每一个最好的分类器都被用了不同的初始化训练了5次；
* matrices是每个分类器的平均准确率和标准差；
* hidden layer sizes从$\{32,64,128,256\} \times \{32,64,128,256\}$选择的
* 正则化权重范围$[10^{-7},10^0]$
* 所有模型的hidden layers都使用ReLU作为激活函数
* 所有模型的优化器都是Adam
* 训练的iterations的最大值都在1000
* scikit-learn v0.22

分类器探针旨在衡量 [基于上下文的representation] 如何捕捉语言属性。分类性能可以帮助我们评估微调的效果。

> classifier-based probing的局限性：
> 它将representation视为一个黑盒，只关注最终任务的性能，而不能揭示finetune是如何改变空间的基本几何结构的。
> 为什么要引入DIRECTPROBE，因为这个prob tech是从几何角度分析embedding的技术。

### （2）DIRECTPROBE
对于给定label的任务，DIRECTPROBE是把具有相同label的points聚成簇，然后返回簇。
![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-10-06-28.png)
无论是左图还是右图，决策边界都必须穿过这些簇之间的间隔区域。
左图：一个简单的二元分类任务，虚线是决策边界
右图：DIRECTPROBE的结果图（之一），灰色是“必须穿过的这些簇之间的间隔区域”，连接起来的点是DIRECTPROBE生成的簇。

finetune的时候，不同的任务下，同一个词基于上下文的representation会有不同的表示，因此，有必要在**给定任务**的前提下，probe这些representation。

得到了这些簇之后，就可以测量这些簇的属性，比如下面这三个：
1. Number of Clusters
   (1) 簇的数量 = label的数量
      简单的**线性**多任务**分类器**
   (2) 簇的数量 > label的数量
      就说明至少有两个同label的样本没有聚到一个簇里，那就需要**非线性分类器**

2. Distances between Clusters
   簇之间的距离揭示了representation的内部结构，通过跟踪fine-tuning期间，这些距离的变化，可以研究representation是怎么变化的。
   为了计算这些距离，本文基于定理fact：一个簇代表一个凸对象a convex object。这样就可以使用max-margin separators最大边距分隔符来计算距离。
   本文训练了一个线性SVM来找到max-margin分隔符并计算它的margin。簇之间的距离=2*margin。


3. Spatial Similarity
   簇之间的距离也可以揭示2个representation的空间相似性。
   如果两个representation在簇之间有相似的相对距离，那么对于当前的任务，这2个representations是相似的。

用这些distance组成一个distance vector $v$,把它当作representation，$v_i$是一对label的簇们之间的距离。
一个任务有n个label:
(1) 当数据集在这个representation下线性可分离，也就是说**簇的数量 = label的数量**,那么：
$$size(v)=\frac{n(n-1)}{2}$$

PS.本文研究的大多数representation都是这种情况。

(2) 非线性可分的情况好像没讨论 ？
 

对于一个带label的任务，还可以计算两个representation的距离向量之间的皮尔逊相关系数，把这个作为两个representation的一种相似性度量。
另外，这个系数也可以用来衡量两个有标签的数据集在相同representation时的相似性。本文利用这一观察结果来分析训练集和测试集在fine-tuned representation下区别。

## 实验设置

### 3.1 Representation
![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-11-33-17.png)
这些模型basic架构是相同的，但是能力不同（比如不同的layers和hidden size）。它们都是基于英文文本&uncased。
对于那些被tokenizer分解为subword的tokens，本文把这些token 的representation的subword embedding进行平均。

> HuggingFace v4.2.1
> Pytorch v1.6.0

### 3.2 Tasks
这些任务覆盖①syntactic语义②semantic语法，这两方面的BERT模型的能力。
1. Part-of-speech tagging (POS)词性标注
   这个任务帮助我们理解是否这个representation捕捉了**粗粒度**的语义分类
2. Dependency relation (DEP)
   预测两个tokens之间的语义依赖关系。
   这个任务帮助我们理解这个representation是否可以描述words之间的语义关系，以及能描述到什么程度。
   这个任务中，需要给一对tokens分配一个类别。具体来说就是，把两个token的（BERT生成的）基于上下文的representation连接起来，这个视为“对”的representation。
   数据集同POS。
3. Preposition supersense disambiguation介词超义消岐
   分类任务，为了消除介词的语法含义的歧义。本文仅在Streusle v4.2 corpus的single-token介词上训练和evaluate。

   1. 预测介词的语义角色semantic role(PS-role)
  
   2. 预测介词的语义功能semantic function(PSfxn)

4. Text classification
   TREC50数据集，每个句子都有50个语法label。
   用的是[CLS]token作为句子的representation。
   这个任务展示了representation表征一个句子的能力。



### 3.3 Fine-tuning Setup
分别单独在上面提过的5个任务上fine-tune那些models。fine-tuned之后的模型生成基于上下文的representation。

初步实验：
  通常，对于BERT_tiny来说，3-5个epoch的fine-tune不够，这些小的representation需要更多的epoch。
  除了$BERT_{base}$，其他的models都用10epochs来fine-tune；$BERT_{base}$用3epochs。
  PS.fine-tune阶段和用于probing的分类器训练阶段分开的，probing classifier是在原始representation和fine-tuned之后的representation之上从头开始train一个2layer神经网络，确保比较是公平的。

## 发现和分析
1. 用classifier去probing是否fine-tuning总是提高分类器的性能。(§4.1)
2. 用DIRECTPROBE给出了一个几何解释，解释为什么fine-tuning提高分类器性能了。(§4.2 and §4.3)
3. 用跨任务fine-tune确认几何解释。(§4.4)
4. 分析fine-tuning是如何改变BERT_base的不同层的几何形状的。(§4.5)

### 4.1 Fine-tuned的性能
![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-15-32-23.png)
上面这个表是BERT_small的结果，tuned指的是在模型最后一层的基础上fine-tuned的结果。最后一列是训练集和测试集的空间相似度Spatial Similarity。
![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-15-37-26.png)
这个表是上表的完整版。一些条目丢失，因为相似性只能在对给定任务线性可分的表示上计算。

结论①：fine-tuning分散了训练集和测试集
在微调之后，所有的相似性都会降低，这意味着由于微调，训练和测试集会有所不同。在大多数情况下，这种差异不足以降低性能。

结论②：也有例外，fine-tuning损害了性能
BERT_small在PS-fxn任务上，tuned之后的性能下降了，并且训练集和测试集的相似度仅0.44，所以作者猜测或许控制训练集和测试集的相似度可以确保微调是有益的。但不确定需要进一步研究。

### 4.2 Representations的线性
结论①：

结论②：

### 4.3 labels的空间结构

结论①：


结论②：


### 4.4 跨任务fine-tuning


结论①：

结论②：

### 4.5 Layer Behavior

结论①：微调不会任意改变表示，即使对于更高的层也是如此

结论②：通过上下两层的直观比较来分析不同层的变化。这里选的是BERT_base，POS tagging任务。

![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-16-10-03.png)
基于POS标签任务和BERTbase的微调前后标签质心差向量的PCA投影。

![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-16-11-23.png)
基于dependency prediction task和BERTbase的微调前后标签质心之间差异向量的PCA投影。

![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-16-12-14.png)
基于Supersense function task和BERTbase的微调前后标签质心差向量的PCA投影.

![](/assets/img/2023-03-02-PaperNote_微调如何改变BERT/2023-03-03-16-14-08.png)
基于Supersense role task和BERTbase的微调前后标签质心之间差异向量的PCA投影。

图7-10显示了基于BERTbase进行微调前后标签质心差向量的PCA投影。


-----
## Note：
### 1. BERT的预训练和微调，微调和P-tuning

BERT的微调过程中是有反向传播的。微调是在预训练模型的基础上，为了适应下游任务而对所有参数进行调整的过程。反向传播是一种优化算法，用于计算梯度并更新参数。

预训练和微调的区别是：

* 预训练是用大量的未标记数据来训练一个通用的模型，例如BERT，以学习语言的特征和表示。
  预训练：随机初始化一个网络模型的参数，然后用大量的未标记数据来训练模型，直到模型的损失越来越小。将训练好的模型的参数保存下来，作为预训练模型。

* 微调是用少量的标记数据来调整预训练模型的参数，以适应特定的下游任务，例如文本分类、命名实体识别等。
  微调：使用预训练模型的参数作为一个新任务的初始化参数，然后用少量的标记数据来训练模型，根据结果不断进行一些修改。将修改后的模型保存下来，作为微调模型。

P-tuning和微调的区别是：

* P-tuning是一种提示优化方法，它只更新预训练模型中的一些特殊的token（如[unused*]），而不更新整个模型的参数。这些特殊的token可以作为模板来引导模型进行下游任务。
* 微调是一种常用的迁移学习方法，它更新预训练模型中的所有参数，以适应下游任务。微调需要更多的内存和计算资源，而且容易过拟合。


P-tuning更新参数，但只更新一些特殊的token，而不是整个模型的参数。这些特殊的token可以看作是模型的前缀，它们可以影响模型的输出。P-tuning只需要很少的参数来微调，因此可以节省内存和计算资源。P-tuning只更新了一些特殊的token，比如[unused1]～[unused6]，它们可以看作是模型的前缀。这些token可以根据标注数据来学习，从而影响模型的输出。其他的模型参数都是冻结的，不会更新。

### 2. 代码不可复现
因为用了gurabi optimizer，docker容器中的授权很难搞要联系客服

### 3. 什么是凸对象？




## Reference

[^1]:[What do you learn from context? Probing for sentence structure in contextualized word representations](https://readpaper.com/paper/2953369973)
[^2]:[Probing what different NLP tasks teach machines about function word comprehension.](https://readpaper.com/paper/2941666437)
[^3]:[DirectProbe: Studying Representations without Classifiers](https://readpaper.com/paper/3154493564)

[^4]:[What Happens To BERT Embeddings During Fine-tuning?](https://readpaper.com/paper/3103368673)
[^5]:[On the Interplay Between Fine-tuning and Sentence-level Probing for Linguistic Knowledge in Pre-trained Transformers](https://readpaper.com/paper/3092448486)



