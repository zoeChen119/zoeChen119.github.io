---
title: Paper Note:神经图匹配网络
date: 2022-10-08 17:10:56 +/-0800
categories: [NLP,论文笔记]
tags: [nlp,神经图匹配网络,论文笔记]     # TAG names should always be lowercase
---
# Paper Note:神经图匹配网络

《Lattice-BERT: Leveraging Multi-Granularity Representations in Chinese Pre-trained Language Models》  
作者：Lu Chen, Yanbin Zhao, Boer Lv, Lesheng Jin, Zhi Chen, Su Zhu, Kai Yu
上海交通大学人工智能研究所
会议：ACL2020

创新点：
目标模型：
目标任务：

摘要：中文短文匹配通常采用单词序列而不是字符序列来获得更好的性能。然而，中文单词的分割可能是错误的、模糊的或不一致的，从而损害了最终的匹配性能。为了解决这个问题，我们提出了神经图谱匹配网络，一个能够处理多粒度输入信息的新型句子匹配框架。取代字符序列或单一的单词序列，由多个单词分割假设形成的成对的单词格子被用作输入，该模型根据一个周到的图形匹配机制学习图形表示。在两个中文数据集上的实验表明，我们的模型超过了最先进的短文匹配模型。

实验结果：