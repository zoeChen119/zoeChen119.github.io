---
title: 调参技巧合集
categories: [NLP,代码技巧和踩坑]
tags: [nlp,调参技巧]     # TAG names should always be lowercase
---
# 调参Tricks记录
#### 1. 样本不平衡
* 问题：
label标注的0-3四类，0类的比重过大，1类其次，2，3类都很少，怎么使用loss的weight来减轻样本不平衡问题？weight参数该如何设置？

* 思路：
大体的思想应该是对　样本较多的类别，使用较小的权重惩罚；
对于样本少的类别，使用较大的权重惩罚；
如何设置weight阈值？
![](/assets/img/调参tricks/2022-09-22-14-21-18.png)

* Trick：
![](/assets/img/调参tricks/2022-09-22-14-22-03.png)

* 原文链接：https://blog.csdn.net/chumingqian/article/details/126625183