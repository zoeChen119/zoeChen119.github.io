---
title: Paddle写代码踩坑记录
categories: [NLP,代码技巧和踩坑]
tags: [nlp,paddle,代码技巧和踩坑]     # TAG names should always be lowercase
---

# Paddle写代码踩坑记录

1. 由于样本不平衡，指定交叉熵损失函数每个类别的权重
```python
# 由于样本不平衡，指定交叉熵损失函数每个类别的权重M2
# M1.[1/count(0),1/count(1)]
# M2.[max(count)/count(0),max(count)/count(1)]
weight_data = np.array([2.31,1.0]).astype("float32")
weight = paddle.to_tensor(weight_data)

criterion = paddle.nn.loss.CrossEntropyLoss(weight=weight)  # 交叉熵损失函数
```
![](/assets/img/paddle合集/2022-09-22-14-08-49.png)
![](/assets/img/paddle合集/2022-09-22-14-09-30.png)
![](/assets/img/paddle合集/2022-09-22-14-09-59.png)

官方文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/CrossEntropyLoss_cn.html#cn-api-nn-loss-crossentropyloss
参考资料：https://blog.csdn.net/qq_48345413/article/details/117925597
备注：需要注意的是weight_data的类型<font color=Sienna>应该是要和logits的类型一致,label的类型必须是int64</font>。
