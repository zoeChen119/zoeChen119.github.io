---
title: Paper Note:对话生成中用知识增强处理未知实体
date: 2022-09-21 11:19:56 +/-0800
categories: [NLP,论文笔记]
tags: [nlp,知识增强,QA-生成,论文笔记]     # TAG names should always be lowercase
---
# Knowledge Enhanced Fine-Tuning for Better Handling Unseen Entities in Dialogue Generation


摘要：对话生成任务中，预训练模型无法应对未登录实体词，为了解决这个问题，现有方法使用外部知识库来生成合适的回复。在现实场景中，实体可能不包含在知识库中，或者受到知识检索精度的影响。为了解决这个问题，我们不引入知识库作为输入，而是仅根据输入上下文预测知识库中的信息，从而迫使模型学习更好的语义表示。具体来说，在知识库的帮助下，我们引入了两个辅助训练目标:
1)(Interpret mask Word)，在给定的语境中推测mask词的含义;
2)(Hypernym Generation)，根据上下文来预测实体的上位词。
在两个对话语料库上的实验结果验证了该方法在知识可用和不可用两种情况下的有效性。