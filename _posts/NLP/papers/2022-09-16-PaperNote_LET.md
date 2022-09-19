---
title: Paper Note:语义学优化中文NLP任务
date: 2022-09-16 10:10:56 +/-0800
categories: [NLP,papers]
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

## <center><font color=Tomato> Question </font></center>

1. 消融实验怎么做的？语义信息和多粒度信息的意义如何证明？  
2. the word lattice graph（词格图）是什么？还有什么方法能够表征“多粒度信息”？
   
  
