---
title: KEPLER：知识嵌入和预训练语言表示的统一模型
categories: [NLP,论文笔记]
tags: [nlp,知识图谱,知识表征,预训练模型,论文笔记]     # TAG names should always be lowercase
---

# KEPLER：知识嵌入和预训练语言表示的统一模型

## Figures

### 1. 首图
这是论文的首图，示范了给知识图谱中每个实体增加描述的效果。

![](E:\Zoe\zoeChen119.github.io/assets/img/2023-02-13-PaperNote_KEPLER/2023-02-13-10-29-28.png)



### 2. （MODEL）图一

![](E:\Zoe\zoeChen119.github.io/assets/img/2023-02-13-PaperNote_KEPLER/2023-02-13-11-34-14.png)

#### 一句话概括
KEPLER通过“*联合训练两个目标objectives*”来把事实知识**隐式融合**进语言表征中。

#### 组件1：Encoder
![](E:\Zoe\zoeChen119.github.io/assets/img/2023-02-13-PaperNote_KEPLER/2023-02-13-14-41-58.png)
每个Token经过L层hiddenlayers的词表征：
$$H_i∈ \Bbb{R}^{N×d},1≤i≤L$$

$$H_i = E_i(H_{i-1})$$

句子表征：
$$E_{<s>}(\cdot)$$

PS:Encoder这部分本文没有调整Transformer编码器结构，没有增加额外的实体链接或者知识融合层。
这样没有额外的推理开销，应用到下游任务也可以像RoBERTa一样用。

#### 组件2：知识Embedding

##### 一句话概括这个子组件
在预训练中采用**知识嵌入目标（KE Objective）**，把知识整合进去。具体来说：没有使用固定的嵌入（存储好的嵌入），而是使用他们对应的文本将实体编码为向量。


> 背景知识：在传统的 KE 模型中，每个实体和关系都被分配了一个 d 维向量，并定义了一个评分函数来训练嵌入和预测链接。

通过选择**不同的文本描述**&**不同的KE评分函数**得到了多种KE-objective选择方案（KEPLER的多种变体）。

(1) 嵌入=实体描述
对于一个三元组$(h,r,t)$:
$$h=E_{<s>}(text_h)$$

$$r=T_r$$

$$t=E_{<s>}(text_t)$$

注释：
1. $text_{h/t}$指头节点/尾节点的那个实体词的文本描述。
2. $<s>$是指序列第一个的那个句表征
3. $T \in \Bbb{R}^{|\cal R| \times d} $ 是存储的关系嵌入，每一个都是d维

KE objective/损失函数[^方法1的损失函数]：
（负采样作为有效优化）
$${\cal{L}}_{KE}=-log \sigma (\gamma - d_r({\bf h,t})) - \sum_{i=1}^{n} \frac{1}{n} log\sigma(d_r( {\bf h_i^{'},t_i^{'}} )-\gamma)$$

注释：
1. $(h_i^{'},r,t_i^{'})$是负样本
2. $\gamma$是margin
3. $\sigma$是sigmoid函数
4. $d_r$是打分函数，打分函数是按照TransE的打分函数，因为简单

$$d_r {({\bf h,t})} = \lVert {\bf h+r-t} \rVert _{\it p}$$

注释：
1. 范数$\it p$=1
2. 负采样策略是固定头部实体，随机抽样尾部实体，反之亦然。


(2) 嵌入=实体和关系描述






(3) 嵌入=以关系为条件的实体嵌入



#### 组件3：MLM objective
仅使用KE objective训练可能会造成灾难性遗忘（本文实验证明的确会差），因此保留MLM objective作为训练目标之一。

使用$RoBERTa_{BASE}$作为预训练初始的checkpoint。



#### 组件4：训练Objectives
设计了一个多任务loss把事实知识和语言理解整合到一个PLM中。
> 文中说到：把两个损失**联合优化**可以**隐式整合**外部KG的知识到the text encoder中，并且同时保持了PLM的语言学理解能力和语义学理解力。

$${\cal L = L}_{KE}+{\cal L}_{MLM}$$

注释：
1. ${\cal L}_{KE}$是KE objective的损失
2. ${\cal L}_{MLM}$是MLM objective的损失

注意：
（对于每一个mini-batch）这两个任务仅仅共享the text encoder，这两个任务采样的数据不一定一样。
这是因为在 MLM 中看到各种文本（而不仅仅是实体描述）可以帮助模型具有更好的语言理解能力。


## 变量和实现

### 1.变量
7个KEPLER变体来测试方法的有效性
#### (1) KEPLER-Wiki
**【基准模型】**

**KG**：Wikidata5M
**知识表征方式**：Entity Descriptions as Embeddings
**Training Objectives**：
$$\mathcal{L} = \mathcal{L}_{KE} + \mathcal{L}_{MLM}$$

**评价**：大部分任务中表现最好


#### (2) KEPLER-WordNet
**KG**：WordNet3.0（节点是引理和同义词集，边是他们的关系）
**知识表征方式**：Entity Descriptions as Embeddings
**Training Objectives**：
$$\mathcal{L} = \mathcal{L}_{KE} + \mathcal{L}_{MLM}$$

**评价**：直观地说，结合WordNet可以带来词汇知识，从而有利于NLP任务。

**PS**：KnowBert也是用的WordNet3.0,是从nltk2包中提取的


#### (3) KEPLER-W+W
**KG**：Wikidata5M，WordNet
**知识表征方式**：Entity Descriptions as Embeddings
**Training Objectives：**
$$\mathcal{L} = \mathcal{L}_{Wiki} + \mathcal{L}_{WorkNet} + \mathcal{L}_{MLM}$$   
其中$\mathcal{L}_{Wiki}$和$\mathcal{L}_{WorkNet}$分别代表Wikidata5M和WordNet的损失。



#### (4) KEPLER-Rel
**KG**：
**知识表征方式**：Entity and Relation Descriptions as Embeddings
$$\hat{r}=E_{<s>}(text_r)$$

**Training Objectives**：
**评价**：由于Wikidata中的关系描述较短(平均11.7个单词)且同质，将关系描述编码为关系嵌入会导致较差的性能，如第4节所示。

#### (5) KEPLER-Cond
**KG**：
**知识表征方式**：entity embedding conditioned on relation method
$${\bf h}_r=E_{<s>}(text_{h,r})$$
**Training Objectives**：
**评价**：该模型在链接预测任务中取得了优异的结果，transductive and inductive


#### (6) KEPLER-OnlyDesc
直接在KE objective的实体描述上训练MLM objective，而不是使用英文维基百科和BookCorpus作为KEPLER的其他版本。
实体描述数据仅2.3GB，并且是同质的

评价：损害一般的语言理解能力，因此表现更差(章节4.2)。




#### (7) KEPLER-KE
只采用了KE目标，这是一个经过删减的KEPLER-Wiki。它用来表明MLM目标对于语言理解的必要性。



### 2.预训练实现
![](E:\Zoe\zoeChen119.github.io/assets/img/2023-02-13-PaperNote_KEPLER/2023-02-13-16-52-36.png)


### 3.微调实现

用不同的NLP任务和KE任务作为评估模型效果的下游任务

1. NLP任务

(1) 关系分类
给定2个实体，判断它们之间的关系
benchmark：TACRED和FewRel
2个框架：Proto + PAIR

(2) 实体分类Entity Typing
给定mention，把它分类到预定义的类型中。
benchmark：OpenEntity


(3) GLUE
一般来说，解决GLUE不需要事实知识（Zhang等人，2019），我们使用它来检查KEPLER是否损害了一般语言理解能力。
结论：这表明，在结合事实知识的同时，KEPLER保持了强大的语言理解能力。然而，KEPLER OnlyDesc的性能显著下降，这表明小规模实体描述数据不足以用MLM训练KEPLER。

2. KE任务

(1) 实验设置

链路预测：
    KEPLER获得实体（方程1）和关系（方程4）嵌入。
    评估方法：3.3节讲过
    baseline：RoBERTa和Our RoBERTa
    打分函数：方程3

In the transductive setting：
    本文的方法和TransE做对比，维度=512，负采样size=64，batch size=2048，学习率=0.001（通过超参数检索得到）
    **负采样大小对KE任务的性能至关重要，但受模型复杂性的限制，KEPLER只能采用1的负采样大小**。


(2)transductive setting



(3)Inductive Setting






## 新数据集：Wikidata5M


## Reference：
[^方法1的损失函数]:https://openreview.net/pdf?id=HkgEQnRqYQ