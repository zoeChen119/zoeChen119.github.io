---
title: Paper Note:语义学优化中文NLP任务
date: 2022-09-16 10:10:56 +/-0800
categories: [NLP,论文笔记]
tags: [nlp,语言学,论文笔记]     # TAG names should always be lowercase
---

# Paper Note:在中文语义表征中引入笔划信息

《cw2vec: Learning Chinese Word Embeddings with Stroke n-gram Information》  
作者：Shaosheng Cao,and Wei Lu,Jun Zhou,Xiaolong Li 
蚂蚁金服 AI Department  
会议：AAAI-2018

创新点：在中文语义表征中引入笔划信息
目标任务：语义表征

摘要：
基于词语和字符的语义表征有2个问题有待商榷：（1）词语和字符的这些信息是否足以正确地捕捉单词的语义信息？（2）是否存在其他有用的信息，可以从单词和字符中提取出来，以更好地建模单词的语义？

由于组成中文<u>词语</u>的<u>字符</u>个数 < 组成英文<u>词语</u>的<u>字符</u>个数，因此，中文的<u>字</u>包含非常丰富的**语义信息**。如果只考虑字符级别的信息，这两个词之间就没有共享的信息，因为它们由不同的字符组成。
比如：木材=木+材；森林=森+林
木，材，森，林单个字没有共享的信息存在，所以仅字符级会损失很多信息。

cw2vec，一种学习中文词嵌入的新方法。利用<font color=DeepSkyBlue>笔画</font>的信息对于改善中文词嵌入。

实验结果：
| Task | Benchmark | Performance |
| --- | --- | --- |
| 任务1. 词语相似度 | ρ × 100 |
