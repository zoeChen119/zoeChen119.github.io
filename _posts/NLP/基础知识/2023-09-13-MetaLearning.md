# Meta-Learning

## 1. Why need Meta-Learning？

### 1.1 Background
大模型在具体的工业落地中，往往需要针对每一个数据集进行训练，训练的目标是找到一个可以拟合**当前数据集**的函数。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-14-52-56.png)
每次都要训练，实在麻烦，那么有没有办法可以找到一个**用少量样本**即可拟合不同领域**所有分类数据集**的函数。

### 1.2. What can meta-learning do?
#### 1.2.1. PLM V.S. PLM + Meta Learning 
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-14-59-51.png)
如图，在Task-Oriented Semantic Parsing任务中，加持了元学习（Reptile方法）之后BART的准确率恒定提升。

#### 1.2.2. MT-DNN V.S. Meta Learning
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-15-00-42.png)
如图，当训练数据越来越少，甚至少样本时，BERT的性能下降明显接近50%，MT-DNN比较稳固，但不如元学习（Reptile方法）的性能，这也佐证了元学习更适合少样本的结论。

#### 1.2.3. Knowledge Distill V.S. Meta Learning
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-15-06-48.png)
知识蒸馏中，有多项研究表明Teacher-Net总是能自己学的很好，但教不会Student Net，因此能否让教师网络“learn to teach”？Meta Learning可以！

#### 1.2.4. Transfer Learning/Fine-tune V.S. Meta Learning
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-15-14-00.png)
其实元学习和迁移学习/微调/多任务学习的界限挺模糊的，思想上很不一样，但实际上做起来好像不太好说。
比如，下面stackexchange上一位答友的[回答](https://ai.stackexchange.com/questions/18232/what-are-the-differences-between-transfer-learning-and-meta-learning)，元学习是指“学会学习”，要学会的东西是一些更高阶的‘元知识’（超参数、初始参数等），就是你训练神经网络的工作；迁移学习是指固定一些层，剩下的层替换成新密基层，来新任务时调整新密集层的参数，在新数据集$B$上重新训练新模型。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-15-27-46.png)
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-28-09.png)

## 2. Meta-Learning Definition
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-30-36.png)
机器学习是先人为调参，之后直接训练特定任务下深度模型。元学习则是先通过其它的任务训练出一个较好的超参数，然后再对特定任务进行训练。
其实就是所有在训练模型时人工设置的超参数都是元学习的目标。另一方面，元学习因为训练过程和机器学习不同，因此元学习和机器学习的数据集构造方式不一样，具体如下，在机器学习中，训练单位是样本数据，通过数据来对模型进行优化；数据可以分为训练集、测试集和验证集。在元学习中，训练单位是任务，一般有两个任务分别是训练任务（Train Tasks）亦称跨任务（Across Tasks）和测试任务（Test Task）亦称单任务（Within Task）。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-33-42.png)

## 3. Meta-Learning Application
> Meta learning是一个通用性的方法论，Meta Learning就等价于汽车中的涡轮增压，可以应用到各种发动机中。

### 3.1. Cross-Domain Training
𝒯_𝑡𝑟𝑎𝑖𝑛 和 𝒯_𝑡𝑒𝑠𝑡属于同一个NLP 问题。比如都是分类数据集。
𝒯_𝑛是不同领域，比如说𝒯_1 是通用领域文本分类数据， 𝒯_2是经济领域文本分类数据。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-38-17.png)

### 3.2. Cross-Question Training
𝒯_𝑡𝑟𝑎𝑖𝑛 和 𝒯_𝑡𝑒𝑠𝑡属于同一领域（或相似领域）不同的NLP 问题。
𝒯_𝑛是不同问题，比如𝒯_𝑡𝑟𝑎𝑖𝑛用的是机器翻译任务和NLI任务，那么𝒯_𝑡𝑒𝑠𝑡 用的是QA和对话状态追踪（DST）。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-39-53.png)

### 3.3. Domain Generalization
需要和跨领域训练cross-domain training区分开。Domain Generalization和Cross-Domain Training的区别也在于各个任务的数据集构造上，应用了交叉构造数据集的方法。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-41-05.png)

## 4. Meta-Learning in NLP

### 4.1. Learning to initialize
通过学习一个好的初始参数来进行快速适应新任务的方法都可以归为 learn-to-init 。MAML及其一阶近似算法（FO-MAML，Reptile，etc.）
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-44-10.png)
这个过程可以看作是构建一个适用于多个目标领域任务的内部表征，或者最大化新任务损失函数对于模型参数的敏感度。

#### 4.1.1. 利用元网络(Meta-Network)来生成一个好的初始参数
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-45-32.png)
该方法强调的是“生成”。
利用元网络(Meta-Network)中的F()就是元网络，它可以根据任务数据生成初始参数，但需要针对不同的模型和任务设计不同的编码器和解码器。

#### 4.1.2. 直接用元模型(Meta-Model)来学习一个好的初始参数（MAML）
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-47-16.png)
元模型就是指在元学习阶段被训练的模型，它可以是任何基于梯度下降算法进行训练的模型，比如CNN、LSTM、RNN及MLP等。
MAML中的F()就是元模型本身，它可以适用于任何深度学习模型和任务类型，但需要计算二阶梯度或使用一阶近似。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-49-12.png)

#### 4.1.3. Learning to initialize V.S. Self-supervised Learning
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-52-09.png)
Learning to initialize和Self-supervised Learning的区别是一个训练时带label，一个不带。

#### 4.1.4. Learning to Initialize v.s. Multi-task Learning
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-53-31.png)
元学习会训练一个通用的模型参数，也就是你的神经网络的初始值。当你遇到一个新的任务时，只需要用少量的样本快速适应（fast adaptation）就可以在新任务上达到很好的效果。
多任务学习会训练一个特定的网络结构，也就是你的神经网络的形式和组成。当你遇到一个新的任务时，比如识别某个数据集中的图像，你会根据这个任务和其他任务之间的关系来决定哪些参数或层要共享，哪些要分离。

### 4.2. Learning to Compare
通过比较任务之间的关系来进行分类。
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-55-00.png)
* MAML中的F()就是元模型本身，它可以适用于任何深度学习模型和任务类型，但需要计算二阶梯度或使用一阶近似。
* learn to initialize中的F()就是元网络，它可以根据任务数据生成初始参数，但需要针对不同的模型和任务设计不同的编码器和解码器。
* learn to compare中的F()则与其他两种方法有本质上的区别，它更像是一个分类器而不是一个元函数。
缺陷：
1. learn to compare方法是一种基于已知分类的方法，它只能从支持集中已有的类别进行分类。
2. learn to compare方法需要对每个查询样本与所有支持集中的样本进行比较，这可能会导致计算量很大，尤其是在支持集较大或查询集较多的情况下。
3. learn to compare方法只考虑了单个查询样本与单个支持集样本之间的相似度，而没有考虑整个查询集与整个支持集之间的全局信息。
4. learn to compare方法依赖于一个有效的相似度计算模块，它需要能够捕捉不同任务或类别之间的语义或逻辑关系。然而，这种相似度计算模块可能很难设计或训练，尤其是在一些复杂或多样化的领域中。

## 5. Meta-Learning in Specific Domain
呼应开头的第一张图，现在我们的目标是训练一个优秀的元学习算法！~
![](E:/Zoe/zoeChen119.github.io/assets/img/2023-09-13-MetaLearning/2023-09-13-16-59-12.png)

## Reference
论文：
1. Lee, H.-Y., Li, S.-W., Vu, N., n.d. Meta Learning for Natural Language Processing: A Survey.
2. Yue, Z., Zeng, H., Zhang, Y., Shang, L., Wang, D., 2023. MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning.
3. Lu, M., Huang, Z., Zhao, Y., Tian, Z., Li, Y., 2023. DaMSTF: Domain Adversarial Learning Enhanced Meta Self-Training for Domain Adaptation.
4. Qin, C., Joty, S., Li, Q., Zhao, R., 2023. Learning to Initialize: Can Meta Learning Improve Cross-task Generalization in Prompt Tuning?
5. Antoniou, A., Edwards, H., Storkey, A., 2018. How to train your MAML. International Conference on Learning Representations,International Conference on Learning Representations.
6. Sun, Q., Liu, Y., Chua, T.-S., Schiele, B., 2019. Meta-Transfer Learning for Few-Shot Learning., in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2019.00049
7. Behl, H., Baydin, A., Torr, PhilipH.S., 2019. Alpha MAML: Adaptive Model-Agnostic Meta-Learning. Cornell University - arXiv,Cornell University - arXiv.
8. Liu, Z., Zhang, R., Song, Y., Zhang, M., 2020. When does MAML Work the Best? An Empirical Study on Model-Agnostic Meta-Learning in NLP Applications. Cornell University - arXiv,Cornell University - arXiv.

课程：
[Meta Learning –Hung-yi Lee- YouTube](https://www.youtube.com/watch?v=EkAqYbpCYAc&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=32)
[ML 2021 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)
[ML 2022 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)

博客：
1. [Few-shot Learning（五）Learning to Compare: Relation Network for Few-Shot Learning - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/614026548)
2. [论文解读（MetaAdapt）《MetaAdapt: Domain Adaptive Few-Shot Misinformation Detection via Meta Learning》 - Wechat~Y466551 - 博客园 (cnblogs.com)](https://www.cnblogs.com/BlairGrowing/p/17652322.html)
3. [What are the differences between transfer learning and meta learning? - Artificial Intelligence Stack Exchange](https://ai.stackexchange.com/questions/18232/what-are-the-differences-between-transfer-learning-and-meta-learning)
4. [Meta-Learning (fastforwardlabs.com)](https://meta-learning.fastforwardlabs.com/#why-should-we-care%3F)
5. [Meta-Learning: Learning to Learn. Although artificial intelligence and… | by Thomas HARTMANN | DataThings | Medium](https://medium.com/datathings/meta-learning-learning-to-learn-a55cadd32b17)

